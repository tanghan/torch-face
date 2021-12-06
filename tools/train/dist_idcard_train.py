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
from core.dataset.baseline_dataset import DataLoaderX
from core.dataset.idcard_dataset import MXIDCardDataset

from core.modules.tail.partial_fc import PartialFC
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging
from core.modules.loss.SST_Prototype import SST_Prototype

import mpi4py.MPI as MPI

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run_train(args, device_id, rank, world_size):
    torch.cuda.set_device(rank)
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    backbone_lr_ratio = args.backbone_lr_ratio
    probe_weights_path = args.probe_weights_path
    gallery_weights_path = args.gallery_weights_path
    loss_type = args.loss_type
    resume = args.resume

    output_dir = args.output_dir
    assert os.path.isdir(output_dir)
    if args.resume:
        assert probe_weights_path is not None
        assert os.path.exists(probe_weights_path)

    train_loader, train_sampler, num_samples, num_classes = get_dataloader(args.rec_path, args.idx_path, rank, batch_size=batch_size, origin_prepro=args.origin_prepro)

    init_logging(rank, output_dir)

    num_images = num_samples
    num_epoch = 20

    total_step = num_images // (batch_size * num_epoch * world_size)
    print("num samples: {}, num classes: {}, total step: {}, num epoch: {}, batch_size: {}, sample_rate: {}, backbone lr ratio: {}, loss type: {}, resume: {}".format(num_samples,
        num_classes, total_step, num_epoch, batch_size, sample_rate, backbone_lr_ratio, loss_type, resume))

    trainer = Trainer(rank, world_size, num_classes=num_classes, num_images=num_images, batch_size=batch_size, num_epoch=num_epoch, sample_rate=sample_rate, probe_weights_path=probe_weights_path, gallery_weights_path=gallery_weights_path, 
            backbone_lr_ratio=backbone_lr_ratio, resume=resume, loss_type=loss_type)
    callback_logging = CallBackLogging(50, rank, total_step, batch_size, world_size, None)
    callback_checkpoint_probe = CallBackModelCheckpoint(rank, output_dir)
    callback_checkpoint_gallery = CallBackModelCheckpoint(rank, output_dir)

    global_step = 0 
    fp16=True

    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    loss = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss().to("cuda:{}".format(rank))
    for epoch in range(0, num_epoch):
        train_sampler.set_epoch(epoch)

        total_step = trainer.train(epoch, global_step, train_loader, grad_amp, loss, criterion, callback_logging, callback_checkpoint_probe, callback_checkpoint_gallery) 
        global_step = total_step

def get_dataloader(rec_path, idx_path, local_rank, batch_size=128, origin_prepro=False):
    train_set = MXIDCardDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=origin_prepro)
    num_samples = len(train_set)
    num_classes = len(train_set.pair_identity)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, train_sampler, num_samples, num_classes


class Trainer():

    def __init__(self, local_rank, world_size, num_classes, num_images, batch_size=128, emb_size=512, num_epoch=12, 
            sample_rate=0.1, resume=True, probe_weights_path="./", gallery_weights_path=None, backbone_lr_ratio=1., loss_type="cosface"):
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.world_size
        self.num_classes = num_classes
        self.emb_size = emb_size

        self.device = "cuda:{}".format(local_rank)
        self.probe_backbone = None
        self.gallery_backbone = None
        self.backbone = None
        self.module_partial_fc = None
        self.num_images = num_images
        warmup_epoch = -1
        self.num_epoch = num_epoch
        self.fp16 = True
        self.warmup_step = self.num_images // self.total_batch_size * warmup_epoch
        self.total_step = self.num_images // self.total_batch_size * self.num_epoch
        #self.decay_epoch = [30, 45, 55, 60, 65, 70]
        self.decay_epoch = [8, 12, 15, 18]
        #self.decay_epoch = [6, 8, 10, 11]
        self.sample_rate = sample_rate
        self.probe_weights_path = probe_weights_path
        self.gallery_weights_path = gallery_weights_path

        self.backbone_lr_ratio = backbone_lr_ratio
        self.resume = resume
        self.loss_type = loss_type
        self.head_conf = "config/head_conf.yaml"
        self.prototype = SST_Prototype(local_rank=local_rank, world_size=world_size, batch_size=batch_size, feat_dim=emb_size,
                queue_size=16384, scale=32.0, loss_type="am_softmax", margin=0.35)

        self.prepare()
        self.alpha = 0.999

    def moving_average(self, alpha):
        """Update the gallery-set network in the momentum way.(MoCo)
        """
        for param_probe, param_gallery in zip(self.probe_backbone.parameters(), self.gallery_backbone.parameters()):
            param_gallery.data =  \
                alpha* param_gallery.data + (1 - alpha) * param_probe.detach().data

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]



    def network_init(self):
        self.probe_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)
        self.gallery_backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)

        if self.resume:
            self.probe_backbone.load_state_dict(torch.load(self.probe_weights_path, map_location=torch.device(self.local_rank)))
            if self.gallery_weights_path is not None:
                self.gallery_backbone.load_state_dict(torch.load(self.gallery_weights_path, map_location=torch.device(self.local_rank)))

            logging.info("resume probe network from {}, gallery network from {}".format(self.probe_weights_path, self.gallery_weights_path))

        self.probe_backbone.to(self.device)
        self.probe_backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.probe_backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.probe_backbone.train()

        self.gallery_backbone.to(self.device)
        self.gallery_backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.gallery_backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.gallery_backbone.train()
        logging.info("init network at {} finished".format(self.local_rank))

    def set_optimizer(self, lr):
        def lr_step_func(current_step):
            decay_step = [x * self.num_images // self.total_batch_size for x in self.decay_epoch]
            if current_step < self.warmup_step:
                return current_step / self.warmup_step
            else:
                return 0.1 ** len([m for m in decay_step if m <= current_step])

        self.opt_backbone = torch.optim.SGD(
            params=[{'params': self.probe_backbone.parameters()}],
            lr=lr / 512 * self.batch_size * self.world_size * self.backbone_lr_ratio,
            momentum=0.9, weight_decay=5e-4)

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone, lr_lambda=lr_step_func)

    def prepare(self):
        self.network_init()
        self.set_optimizer(lr=0.1)

    def train(self, epoch, global_step, train_loader, grad_amp, loss,
            criterion, callback_logging, callback_checkpoint_probe, callback_checkpoint_gallery):
        for step, (probe_imgs, gallery_imgs, labels) in enumerate(train_loader):
            #probe_label, gallery_label = torch.split(labels, 2, 1)

            probe_features1 = F.normalize(self.probe_backbone(probe_imgs))
            probe_features2 = F.normalize(self.probe_backbone(gallery_imgs))

            with torch.no_grad():
                probe_imgs, probe_idx_unshuffle = self._batch_shuffle_ddp(probe_imgs)
                gallery_features1 = F.normalize(self.gallery_backbone(probe_imgs))

                gallery_imgs, gallery_idx_unshuffle = self._batch_shuffle_ddp(gallery_imgs)
                gallery_features2 = F.normalize(self.gallery_backbone(gallery_imgs))

                gallery_features1 = self._batch_unshuffle_ddp(gallery_features1, probe_idx_unshuffle)
                gallery_features2 = self._batch_unshuffle_ddp(gallery_features2, gallery_idx_unshuffle)

            output1, output2, label, id_set  = self.prototype(
                probe_features1, gallery_features2, probe_features2, gallery_features1, labels)

            loss_v = (criterion(output1, label) + criterion(output2, label))/2

            if self.fp16:
                grad_amp.scale(loss_v).backward()
                grad_amp.unscale_(self.opt_backbone)
                clip_grad_norm_(self.probe_backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(self.opt_backbone)
                grad_amp.update()
            else:
                loss_v.backward()
                clip_grad_norm_(self.probe_backbone.parameters(), max_norm=5, norm_type=2)
                self.opt_backbone.step()

            self.opt_backbone.zero_grad()
            loss.update(loss_v, 1)
            self.moving_average(self.alpha)
            callback_logging(step + global_step, loss, epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
            self.scheduler_backbone.step()

        total_step = global_step + step
        callback_checkpoint_probe(total_step, self.probe_backbone, None, "probe-")
        callback_checkpoint_gallery(total_step, self.gallery_backbone, None, "gallery-")
        #save_gallery_backbone_path = "gallery-backbone.pth"
        #if self.local_rank == 0:
        #    torch.save(self.gallery_backbone.state_dict(), save_gallery_backbone_path)

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
    parser.add_argument("--probe_weights_path", type=str, default=None, help="")
    parser.add_argument("--gallery_weights_path", type=str, default=None, help="")
    parser.add_argument("--backbone_lr_ratio", type=float, default=1., help="")
    parser.add_argument("--origin_prepro", action="store_true", help="")
    parser.add_argument("--loss_type", type=str, default="arcface", help="")
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    args = parser.parse_args()
    main(args)
