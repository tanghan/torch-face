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

from core.modules.tail.dist_partial_softmax import DistPartialSoftmax
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging
import mpi4py.MPI as MPI

class StudentBackbone(nn.Module):

    def __init__(self, local_rank, world_size, s_emb_size, t_emb_size, fp16=True, resume=False, pretrain_path="./"):
        super(StudentBackbone, self).__init__()
        self.out = None
        self.local_rank = local_rank
        self.world_size = world_size
        self.fp16 = fp16
        self.out = None
        self.s_backbone = None
        self.s_emb_size = s_emb_size
        self.t_emb_size = t_emb_size
        self.resume = resume
        self.pretrain_path = None
        if self.resume:
            self.pretrain_path = pretrain_path


        self.build_output_layer()
        self.build_backbone()


    def build_output_layer(self):
        self.out = nn.Sequential(
                nn.Linear(self.s_emb_size, self.t_emb_size),
                nn.BatchNorm1d(self.t_emb_size)
                )

    def build_backbone(self):
        self.s_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.s_emb_size)
        if self.resume:
            self.s_backbone.load_state_dict(torch.load(self.pretrain_path, map_location=torch.device(self.local_rank)))

            logging.info("resume student network from {}".format(self.pretrain_path))


    def forward(self, x):
        logits = self.s_backbone(x)
        kd_logits = self.out(logits)
        return logits, kd_logits

def run_train(args, device_id, local_rank, world_size):
    torch.cuda.set_device(local_rank)
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    backbone_lr_ratio = args.backbone_lr_ratio
    weights_path = args.weights_path
    fc_prefix = args.fc_prefix
    loss_type = args.loss_type
    resume = args.resume
    pretrain_path = args.pretrain_path
    fp16 = args.fp16

    output_dir = args.output_dir
    assert os.path.isdir(output_dir)
    if args.resume:
        assert pretrain_path is not None
        assert fc_prefix is not None
        assert os.path.exists(pretrain_path)

    train_loader, train_sampler, num_samples, num_classes = get_dataloader(args.rec_path, args.idx_path, local_rank, batch_size=batch_size, origin_prepro=args.origin_prepro)

    init_logging(device_id, output_dir)

    num_images = num_samples
    num_epoch = 35

    total_step = num_images // (batch_size * num_epoch * world_size)
    print("num samples: {}, num classes: {}, total step: {}, num epoch: {}, teacher weights path: {}, batch_size: {}, sample_rate: {}, backbone lr ratio: {}, loss type: {}, resume: {}, use fp16: {}, pretrain path: {}".format(num_samples,
        num_classes, total_step, num_epoch, weights_path, batch_size, sample_rate, backbone_lr_ratio, loss_type, resume, fp16, pretrain_path))

    trainer = Trainer(device_id, local_rank, world_size, num_classes=num_classes, num_images=num_images, batch_size=batch_size, num_epoch=num_epoch, sample_rate=sample_rate, weights_path=weights_path, fc_prefix=fc_prefix, backbone_lr_ratio=backbone_lr_ratio, resume=resume, loss_type=loss_type, fp16=fp16, pretrain_path=pretrain_path)
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
            sample_rate=0.1, resume=True, weights_path="./", fc_prefix="./", backbone_lr_ratio=1., loss_type="cosface", fp16=True, pretrain_path="./"):
        self.device_id = device_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.world_size
        self.num_classes = num_classes
        self.resume = resume

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
        #self.decay_epoch = [8, 12, 15, 18]
        self.decay_epoch = [16, 24, 30, 33]
        #self.decay_epoch = [32, 48, 54, 58]
        #self.decay_epoch = [6, 8, 10, 11]
        self.sample_rate = sample_rate
        self.weights_path = weights_path
        self.fc_prefix = fc_prefix
        self.s_backbone = None
        self.t_backbone = None
        self.t_emb_size = t_emb_size
        self.s_emb_size = s_emb_size

        self.backbone_lr_ratio = backbone_lr_ratio
        self.pretrain_path = pretrain_path
        self.loss_type = loss_type
        self.head_conf = "config/head_conf.yaml"
        self.head_factory = HeadFactory(self.local_rank, self.world_size, self.loss_type, self.head_conf)

        self.prepare()

    def network_init(self):
        self.t_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.t_emb_size)

        self.t_backbone.load_state_dict(torch.load(self.weights_path, map_location=torch.device(self.local_rank)))

        logging.info("resume network from {}".format(self.weights_path))
        s_backbone = StudentBackbone(self.local_rank, self.world_size, t_emb_size=self.t_emb_size, s_emb_size=self.s_emb_size, fp16=self.fp16, resume=self.resume, pretrain_path=self.pretrain_path)

        self.t_backbone = self.t_backbone.to(self.device)
        s_backbone = s_backbone.to(self.device)
        self.s_backbone = torch.nn.parallel.DistributedDataParallel(
            module=s_backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.s_backbone.train()
        self.t_backbone.train()
        logging.info("init network at {} finished".format(self.local_rank))


    def set_loss(self):
        #self.loss_fn = get_loss(self.loss_type)
        self.loss_fn = self.head_factory.get_head()

    def set_tail(self, loss_fn):

        self.module_fc = DistPartialSoftmax(
            rank=self.device_id, local_rank=self.local_rank, world_size=self.world_size, resume=self.resume,
            batch_size=self.batch_size, margin_softmax=loss_fn, num_classes=self.num_classes, sample_rate=self.sample_rate,
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
        self.set_optimizer(lr=0.01)

    
    def train(self, epoch, global_step, train_loader, grad_amp, loss_log, kd_loss_log,
            callback_logging, callback_checkpoint):
        for step, (img, label) in enumerate(train_loader):

            with torch.no_grad():
                t_features = self.t_backbone(img)
                t_features = F.normalize(t_features)
            s_features, kd_features = self.s_backbone(img)
            kd_features = F.normalize(kd_features)
            
            loss_v, loss_g = self.module_fc(s_features, label, self.opt_pfc)
            kd_loss = (1. - (t_features * kd_features).sum(dim=1)).mean()
            loss_v += 0.1 * kd_loss
                    
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
            self.module_fc.update()
            self.opt_backbone.zero_grad()
            self.opt_pfc.zero_grad()
            loss_log.update(loss_v.detach(), 1)
            kd_loss_log.update(kd_loss.detach(), 1)
            callback_logging(step + global_step, [loss_log, kd_loss_log], epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
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
    parser.add_argument("--pretrain_path", type=str, default=None, help="")
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    args = parser.parse_args()
    main(args)
