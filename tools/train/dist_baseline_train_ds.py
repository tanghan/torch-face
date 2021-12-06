import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import argparse
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from core.models.resnet import iresnet
from core.modules.loss.losses import get_loss
from core.modules.loss.head_def import HeadFactory
from utils.callback_utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX

from core.modules.tail.dist_softmax import DistSoftmax
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging
import mpi4py.MPI as MPI


def run_train(args, device_id, local_rank, world_size):
    torch.cuda.set_device(local_rank)
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    backbone_lr_ratio = args.backbone_lr_ratio
    weights_path = args.weights_path
    fc_prefix = args.fc_prefix
    loss_type = args.loss_type
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

    trainer = Trainer(device_id, local_rank, world_size, num_classes=num_classes, num_images=num_images, batch_size=batch_size, num_epoch=num_epoch, sample_rate=sample_rate, weights_path=weights_path, fc_prefix=fc_prefix, backbone_lr_ratio=backbone_lr_ratio, resume=resume, loss_type=loss_type, fp16=fp16)
    callback_logging = CallBackLogging(50, device_id, total_step, batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(device_id, output_dir)

    global_step = 0 

    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    loss = AverageMeter()
    for epoch in range(0, num_epoch):
        train_sampler.set_epoch(epoch)
        total_step = trainer.train(epoch, global_step, train_loader, grad_amp, loss, callback_logging, callback_checkpoint) 
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

    def __init__(self, device_id, local_rank, world_size, num_classes, num_images, batch_size=128, emb_size=512, num_epoch=12, 
            sample_rate=0.1, resume=True, weights_path="./", fc_prefix="./", backbone_lr_ratio=1., loss_type="cosface", fp16=True):
        self.device_id = device_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.world_size
        self.num_classes = num_classes
        self.emb_size = emb_size

        self.device = "cuda:{}".format(local_rank)
        self.backbone = None
        self.module_fc = None
        self.loss_fn = None
        self.num_images = num_images
        warmup_epoch = 1
        self.num_epoch = num_epoch
        self.fp16 = fp16
        self.warmup_step = self.num_images // self.total_batch_size * warmup_epoch
        self.total_step = self.num_images // self.total_batch_size * self.num_epoch
        #self.decay_epoch = [30, 45, 55, 60, 65, 70]
        self.decay_epoch = [8, 12, 15, 18]
        #self.decay_epoch = [6, 8, 10, 11]
        self.sample_rate = sample_rate
        self.weights_path = weights_path
        self.fc_prefix = fc_prefix

        self.backbone_lr_ratio = backbone_lr_ratio
        self.resume = resume
        self.loss_type = loss_type
        self.head_conf = "config/head_conf.yaml"
        self.head_factory = HeadFactory(self.local_rank, self.world_size, self.loss_type, self.head_conf)

        self.prepare()

    def network_init(self):
        backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)

        if self.resume:
            backbone.load_state_dict(torch.load(self.weights_path, map_location=torch.device(self.local_rank)))

            logging.info("resume network from {}".format(self.weights_path))

        backbone = backbone.to(self.device)
        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.backbone.train()
        logging.info("init network at {} finished".format(self.local_rank))


    def set_loss(self):
        #self.loss_fn = get_loss(self.loss_type)
        self.loss_fn = self.head_factory.get_head()

    def set_tail(self, loss_fn):

        self.module_fc = DistSoftmax(
            rank=self.device_id, local_rank=self.local_rank, world_size=self.world_size, resume=self.resume,
            batch_size=self.batch_size, margin_softmax=loss_fn, num_classes=self.num_classes,
            embedding_size=self.emb_size)

    def set_optimizer(self, lr):
        def lr_step_func(current_step):
            decay_step = [x * self.num_images // self.total_batch_size for x in self.decay_epoch]
            if current_step < self.warmup_step:
                return current_step / self.warmup_step
            else:
                return 0.1 ** len([m for m in decay_step if m <= current_step])

        self.opt_backbone = torch.optim.SGD(
            params=[{'params': self.backbone.parameters()}],
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

    
    def train(self, epoch, global_step, train_loader, grad_amp, loss,
            callback_logging, callback_checkpoint):
        for step, (img, label) in enumerate(train_loader):

            features = self.backbone(img)

            features = F.normalize(features)
            loss_v, loss_g = self.module_fc(features, label)

            if self.fp16:
                grad_amp.scale(loss_v).backward()
                grad_amp.unscale_(self.opt_backbone)
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(self.opt_backbone)
                grad_amp.update()
            else:
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                self.opt_backbone.step()

            self.opt_pfc.step()
            self.opt_backbone.zero_grad()
            self.opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(step + global_step, loss, epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
            self.scheduler_backbone.step()
            self.scheduler_pfc.step()
        total_step = global_step + step
        if self.device_id == 0:
            callback_checkpoint(total_step, self.backbone, self.module_fc)

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
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    args = parser.parse_args()
    main(args)
