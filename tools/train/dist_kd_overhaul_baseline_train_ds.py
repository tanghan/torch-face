import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import argparse
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn

from core.models.resnet import iresnet
from core.modules.loss.losses import get_loss
from core.modules.loss.head_def import HeadFactory
from utils.callback_utils.utils_callbacks import CallBackVerification, CallBackLoggingList, CallBackModelCheckpoint
from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX

from core.modules.tail.dist_softmax import DistSoftmax
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging
from scipy.stats import norm
import math
import mpi4py.MPI as MPI
import numpy as np

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def get_margin_from_BN(bn, rank=0):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = float(abs(s.item()))
        m = float(m.item())
        if s < 1e-6:
            s = 1e-6
        thresh_1 = -m /s
        thresh = norm.cdf(thresh_1)

        if thresh > 0.001:
            margin_ = - s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m
        else:
            margin_ = -3 * s
        margin.append(margin_)
        #if rank == 0:
        #    print(np.max(np.abs(margin_)))

    return torch.FloatTensor(margin).to(std.device)

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


class StudentBackbone(nn.Module):

    def __init__(self, device_id, local_rank, world_size, s_emb_size, t_emb_size, fp16=True, kd_last=False):
        super(StudentBackbone, self).__init__()
        self.out = None
        self.device_id = device_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.fp16 = fp16
        self.out = None
        self.s_backbone = None
        self.s_emb_size = s_emb_size
        self.t_emb_size = t_emb_size

        t_channels = [64, 128, 256, 512]
        s_channels = [64, 128, 256, 512]
        self.feat_num = len(s_channels)

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        self.build_backbone()
        self.out = None
        if kd_last:
            self.build_output_layer()

    def build_backbone(self):
        self.s_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.s_emb_size)

    def build_output_layer(self):
        self.out = nn.Sequential(
                nn.Linear(self.s_emb_size, self.t_emb_size),
                nn.BatchNorm1d(self.t_emb_size)
                )

    def build_margin(self, teacher_bns):

        margins = [get_margin_from_BN(bn, self.device_id) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())


    def forward(self, x, t_feats):
        feats, logits = self.s_backbone.extract_feature(x)
        loss_distill = 0.
        kd_logits = None
        for i in range(self.feat_num):
            if self.fp16:
                upper_feats = self.Connectors[i](feats[i].float())

                loss_distill_ = distillation_loss(upper_feats, t_feats[i].float().detach(), getattr(self, 'margin%d' % (i+1))) / 2 ** (self.feat_num - i - 1)
            else:
                upper_feats = self.Connectors[i](feats[i])
                loss_distill_ = distillation_loss(upper_feats, t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) / 2 ** (self.feat_num - i - 1)
            loss_distill += loss_distill_
        if self.out is not None:
            kd_logits = self.out(logits.float() if self.fp16 else logits)

        return loss_distill, logits, kd_logits

def run_train(args, device_id, local_rank, world_size):
    torch.cuda.set_device(local_rank)
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    backbone_lr_ratio = args.backbone_lr_ratio
    weights_path = args.weights_path
    fc_prefix = args.fc_prefix
    loss_type = args.loss_type
    kd_last = args.kd_last
    resume = args.resume
    fp16 = args.fp16

    output_dir = args.output_dir
    assert os.path.isdir(output_dir)
    if args.resume:
        assert weights_path is not None
        assert fc_prefix is not None
        assert os.path.exists(weights_path)

    train_loader, train_sampler, num_samples, num_classes = get_dataloader(args.rec_path, args.idx_path, local_rank, batch_size=batch_size, origin_prepro=args.origin_prepro)

    init_logging(device_id, output_dir)

    num_images = num_samples
    num_epoch = 20

    total_step = num_images // (batch_size * num_epoch * world_size)
    print("num samples: {}, num classes: {}, total step: {}, num epoch: {}, batch_size: {}, sample_rate: {}, backbone lr ratio: {}, loss type: {}, resume: {}, use fp16: {}".format(num_samples,
        num_classes, total_step, num_epoch, batch_size, sample_rate, backbone_lr_ratio, loss_type, resume, fp16))

    trainer = Trainer(device_id, local_rank, world_size, num_classes=num_classes, num_images=num_images, batch_size=batch_size, num_epoch=num_epoch, sample_rate=sample_rate, weights_path=weights_path, fc_prefix=fc_prefix, backbone_lr_ratio=backbone_lr_ratio, resume=resume, loss_type=loss_type, fp16=fp16, kd_last=kd_last)
    callback_logging = CallBackLoggingList(50, device_id, total_step, batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(device_id, output_dir)

    global_step = 0 

    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    loss_log = AverageMeter(name="total loss")
    kd_loss_log = AverageMeter(name="kd loss")
    for epoch in range(0, num_epoch):
        train_sampler.set_epoch(epoch)
        total_step = trainer.train(epoch, global_step, train_loader, grad_amp, loss_log, kd_loss_log, callback_logging, callback_checkpoint) 
        global_step = total_step

def get_dataloader(rec_path, idx_path, local_rank, batch_size=128, origin_prepro=False):
    train_set = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=origin_prepro)
    num_samples = len(train_set)
    num_classes = len(train_set.id_seq)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=1234)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, train_sampler, num_samples, num_classes


class Trainer():

    def __init__(self, device_id, local_rank, world_size, num_classes, num_images, batch_size=128, t_emb_size=512, s_emb_size=256, num_epoch=12, 
            sample_rate=0.1, resume=True, weights_path="./", fc_prefix="./", backbone_lr_ratio=1., loss_type="cosface", fp16=True, kd_last=False):
        self.device_id = device_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.world_size
        self.num_classes = num_classes

        self.device = "cuda:{}".format(local_rank)
        self.module_fc = None
        self.loss_fn = None
        self.num_images = num_images
        warmup_epoch = 1
        self.num_epoch = num_epoch
        self.fp16 = fp16
        self.warmup_step = self.num_images // self.total_batch_size * warmup_epoch
        self.total_step = self.num_images // self.total_batch_size * self.num_epoch
        #self.decay_epoch = [30, 45, 55, 60, 65, 70]
        #self.decay_epoch = [8, 12, 15, 18]
        self.decay_epoch = [8, 12, 15, 18]
        #self.decay_epoch = [32, 48, 54, 58]
        #self.decay_epoch = [6, 8, 10, 11]
        self.sample_rate = sample_rate
        self.weights_path = weights_path
        self.fc_prefix = fc_prefix
        self.s_backbone = None
        self.t_backbone = None
        self.t_emb_size = t_emb_size
        self.s_emb_size = s_emb_size
        self.kd_last = kd_last

        self.backbone_lr_ratio = backbone_lr_ratio
        self.loss_type = loss_type
        self.head_conf = "config/head_conf.yaml"
        self.head_factory = HeadFactory(self.local_rank, self.world_size, self.loss_type, self.head_conf)

        self.prepare()

    def network_init(self):
        self.t_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.t_emb_size)

        self.t_backbone.load_state_dict(torch.load(self.weights_path, map_location=torch.device(self.local_rank)))
        self.t_backbone = self.t_backbone.to(self.device)
        self.t_backbone.eval()

        logging.info("resume network from {}".format(self.weights_path))
        s_backbone = StudentBackbone(self.device_id, self.local_rank, self.world_size, t_emb_size=self.t_emb_size, s_emb_size=self.s_emb_size, fp16=self.fp16)
        s_backbone.build_margin(self.t_backbone.get_bn_before_relu())

        s_backbone = s_backbone.to(self.device)
        self.s_backbone = torch.nn.parallel.DistributedDataParallel(
            module=s_backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.s_backbone.train()
        logging.info("init network at {} finished".format(self.local_rank))


    def set_loss(self):
        #self.loss_fn = get_loss(self.loss_type)
        self.loss_fn = self.head_factory.get_head()

    def set_tail(self, loss_fn):

        self.module_fc = DistSoftmax(
            rank=self.device_id, local_rank=self.local_rank, world_size=self.world_size, resume=False,
            batch_size=self.batch_size, margin_softmax=loss_fn, num_classes=self.num_classes,
            embedding_size=self.s_emb_size)

    def set_optimizer(self, lr):
        def lr_step_func(current_step):
            decay_step = [x * self.num_images // self.total_batch_size for x in self.decay_epoch]
            if current_step < self.warmup_step:
                return current_step / self.warmup_step
            else:
                return 0.1 ** len([m for m in decay_step if m <= current_step])

        self.opt_backbone = torch.optim.SGD(
            params=[{'params': self.s_backbone.parameters()}],
            lr=lr / 512 * self.batch_size * self.world_size * self.backbone_lr_ratio,
            momentum=0.9, weight_decay=5e-4)

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone, lr_lambda=lr_step_func)

        self.opt_pfc = torch.optim.SGD(
            params=[{'params': self.module_fc.parameters()}],
            lr=lr / 512 * self.batch_size * self.world_size,
            momentum=0.9, weight_decay=5e-4)
        for p in self.module_fc.parameters():
            print("rank: {}, shape: {}".format(self.device_id, p.shape))
        self.scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_pfc, lr_lambda=lr_step_func)

    def prepare(self):
        self.network_init()
        self.set_loss()
        self.set_tail(loss_fn=self.loss_fn)
        self.set_optimizer(lr=0.1)

    
    def train(self, epoch, global_step, train_loader, grad_amp, loss_log, kd_loss_log,
            callback_logging, callback_checkpoint):
        for step, (img, label) in enumerate(train_loader):

            with torch.no_grad():
                t_feats, t_logits = self.t_backbone.extract_feature(img)
                if self.kd_last:
                    t_logits = F.normalize(t_logits)
            distill_loss, s_logits, kd_logits = self.s_backbone(img, t_feats)
            
            loss_v, loss_g = self.module_fc(s_logits, label)
            if kd_logits is not None:
                kd_features = F.normalize(kd_logits)
                kd_loss = (1. - (t_logits * kd_logits).sum(dim=1).mean())
                loss_v += kd_loss
            distill_loss = distill_loss.sum() / self.world_size / self.batch_size / 10000
            print("rank: {}, distill_loss: {}, loss_v: {}".format(self.device_id, distill_loss, loss_v))
            loss_v += distill_loss
                    
            if loss_g is not None:
                loss_v += loss_v + torch.mean(loss_g) / self.world_size

            if self.fp16:
                grad_amp.scale(loss_v).backward()
                dist.barrier()
                grad_amp.unscale_(self.opt_backbone)
                torch.nn.utils.clip_grad_norm_(self.s_backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(self.opt_backbone)
                grad_amp.update()
            else:
                loss_v.backward()
                dist.barrier()
                torch.nn.utils.clip_grad_norm_(self.s_backbone.parameters(), max_norm=5, norm_type=2)
                self.opt_backbone.step()

            self.opt_pfc.step()
            self.opt_backbone.zero_grad()
            self.opt_pfc.zero_grad()
            loss_log.update(loss_v.detach(), 1)
            if kd_logits is not None:
                kd_loss_log.update(kd_loss.detach() + distill_loss.detach(), 1)
            else:
                kd_loss_log.update(distill_loss.detach(), 1)
            log_list = [loss_log, kd_loss_log]
            callback_logging(step + global_step, log_list, epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
            self.scheduler_backbone.step()
            self.scheduler_pfc.step()
        total_step = global_step + step
        if self.device_id == 0:
            callback_checkpoint(total_step, self.s_backbone, self.module_fc)

        return total_step        


def main(args):
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    world_size = comm.Get_size()
    dist_url = args.dist_url
    gpu_num = args.gpu_num
    print('MPI local_rank=%d, world_size=%d' % (local_rank, world_size))
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank
        )
    except Exception as e:
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    current_device = local_rank % gpu_num
    run_train(args, local_rank, current_device, world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--gpu_num", type=int, default=8, help="")
    parser.add_argument("--rec_path", type=str, default="/cluster_home/data/train_data/id_card/njn/small_njn.rec", help="")
    parser.add_argument("--idx_path", type=str, default="/cluster_home/data/train_data/id_card/njn/small_njn.idx", help="")
    parser.add_argument("--batch_size", type=int, default=128, help="")
    parser.add_argument("--output_dir", type=str, default="/job_data/", help="")
    parser.add_argument("--sample_rate", type=float, default=1., help="")
    parser.add_argument("--resume", action="store_true", help="")
    parser.add_argument("--weights_path", type=str, default=None, help="")
    parser.add_argument("--fc_prefix", type=str, default="./", help="")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.1, help="")
    parser.add_argument("--origin_prepro", action="store_true", help="")
    parser.add_argument("--loss_type", type=str, default="arcface", help="")
    parser.add_argument("--fp16", action="store_true", help="")
    parser.add_argument("--kd_last", action="store_true", help="")
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    args = parser.parse_args()
    main(args)
