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
from utils.callback_utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from core.dataset.baseline_dataset import MXFaceDataset, DataLoaderX

from core.modules.tail.partial_fc import PartialFC
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging

def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run_train(args, rank, world_size):
    torch.cuda.set_device(rank)
    batch_size = args.batch_size
    sample_rate = args.sample_rate
    train_loader, train_sampler = get_dataloader(args.rec_path, args.idx_path, rank, batch_size=batch_size, origin_prepro=args.origin_prepro)
    output_dir = args.output_dir

    init_logging(rank, output_dir)

    num_image = 4800000
    num_epoch = 20
    trainer = Trainer(rank, world_size, batch_size=batch_size, num_epoch=num_epoch, sample_rate=sample_rate, weights_prefix=args.weights_prefix)
    total_step = num_image // (batch_size * num_epoch * world_size)
    callback_logging = CallBackLogging(50, rank, total_step, batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, output_dir)

    global_step = 0 
    fp16=True

    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    loss = AverageMeter()
    for epoch in range(0, num_epoch):
        train_sampler.set_epoch(epoch)
        total_step = trainer.train(epoch, global_step, train_loader, grad_amp, loss, callback_logging, callback_checkpoint) 
        global_step = total_step

def get_dataloader(rec_path, idx_path, local_rank, batch_size=128, origin_prepro=False):
    train_set = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=origin_prepro)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    return train_loader, train_sampler


class Trainer():

    def __init__(self, local_rank, world_size, batch_size=128, emb_size=512, num_epoch=20, sample_rate=0.1, weights_prefix=None):
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_batch_size = self.batch_size * self.world_size
        self.num_classes = 500000
        self.emb_size = emb_size

        self.device = "cuda:{}".format(local_rank)
        self.backbone = None
        self.module_partial_fc = None
        self.loss_fn = None
        self.num_image = 4800000
        warmup_epoch = -1
        self.num_epoch = num_epoch
        self.fp16 = True
        self.warmup_step = self.num_image // self.total_batch_size * warmup_epoch
        self.total_step = self.num_image // self.total_batch_size * self.num_epoch
        #self.decay_epoch = [30, 45, 55, 60, 65, 70]
        self.decay_epoch = [8, 12, 15, 18]
        self.sample_rate = sample_rate
        self.weights_prefix = weights_prefix

        self.prepare()

    def network_init(self):
        self.backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)

        if self.weights_prefix is not None:
            backbone_pth = os.path.join(self.weights_prefix, "backbone.pth")
            self.backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(self.local_rank)))

        self.backbone.to(self.device)
        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        self.backbone.train()
        logging.info("init network at {} finished".format(self.local_rank))


    def set_loss(self):
        self.loss_fn = get_loss("cosface")

    def set_tail(self, loss_fn):

        self.module_partial_fc = PartialFC(
            rank=self.local_rank, local_rank=self.local_rank, world_size=self.world_size, resume=True,
            batch_size=self.batch_size, margin_softmax=loss_fn, num_classes=self.num_classes,
            sample_rate=self.sample_rate, embedding_size=self.emb_size, prefix=self.weights_prefix)

    def set_optimizer(self, lr):
        def lr_step_func(current_step):
            decay_step = [x * self.num_image // self.total_batch_size for x in self.decay_epoch]
            if current_step < self.warmup_step:
                return current_step / self.warmup_step
            else:
                return 0.1 ** len([m for m in decay_step if m <= current_step])

        self.opt_backbone = torch.optim.SGD(
            params=[{'params': self.backbone.parameters()}],
            lr=lr / 512 * self.batch_size * self.world_size,
            momentum=0.9, weight_decay=5e-4)

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone, lr_lambda=lr_step_func)

        self.opt_pfc = torch.optim.SGD(
            params=[{'params': self.module_partial_fc.parameters()}],
            lr=lr / 512 * self.batch_size * self.world_size,
            momentum=0.9, weight_decay=5e-4)
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
            features = F.normalize(self.backbone(img))
            x_grad, loss_v = self.module_partial_fc.forward_backward(label, features, self.opt_pfc)
            if self.fp16:
                features.backward(grad_amp.scale(x_grad))
                grad_amp.unscale_(self.opt_backbone)
                clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(self.opt_backbone)
                grad_amp.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            self.opt_pfc.step()
            self.module_partial_fc.update()
            self.opt_backbone.zero_grad()
            self.opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(step + global_step, loss, epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
            self.scheduler_backbone.step()
            self.scheduler_pfc.step()
        total_step = global_step + step
        callback_checkpoint(total_step, self.backbone, self.module_partial_fc)

        return total_step        


def main(args):
    size = args.gpu_num
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, run_train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--gpu_num", type=int, default=8, help="")
    parser.add_argument("--rec_path", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.rec", help="")
    parser.add_argument("--idx_path", type=str, default="/home/users/han.tang/data/baseline_2030_V0.2/baseline_2030_V0.2.idx", help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--output_dir", type=str, default="/job_data/", help="")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="")
    parser.add_argument("--resume", action="store_true", help="")
    parser.add_argument("--weights_prefix", type=str, default="/home/users/han.tang/workspace/weight_imprint/imprint_weights", help="")
    parser.add_argument("--origin_prepro", action="store_true", help="")
    args = parser.parse_args()
    main(args)
