import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import argparse
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset

from core.models.resnet import iresnet
from core.modules.loss.losses import get_loss
from utils.callback_utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from core.dataset.dataset import MXFaceDataset, DataLoaderX
from core.dataset.sampler.multi_sampler import DistributedMultiBatchSampler

from core.modules.tail.partial_fc import PartialFC
from torch.nn.utils import clip_grad_norm_
from utils.utils_amp import MaxClipGradScaler
from utils.logging_utils.utils_logging import AverageMeter, init_logging

from easydict import EasyDict as edict 
from importlib.machinery import SourceFileLoader


def init_process(rank, world_size, opt, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(opt, rank, world_size)

def run_train(opt, rank, world_size):
    torch.cuda.set_device(rank)
    seed = opt.utils.seed

    trainset = opt.dataset.trainset

    num_samples_list = []
    batch_size_list = []
    num_classes_list = []

    for dataname in trainset:
        num_samples = opt.dataset[dataname].num_samples
        num_classes = opt.dataset[dataname].num_classes
        batch_size = opt.dataset[dataname].batch_size

        num_samples_list.append(num_samples)
        batch_size_list.append(batch_size)
        num_classes_list.append(num_classes)

    train_loader = get_dataloader(rank, opt, seed)
    output_dir = opt.utils.checkpoint 

    init_logging(rank, output_dir)

    num_image = 17091657
    num_epoch = opt.utils.num_epoch
    trainer = Trainer(rank, world_size, num_classes_list, num_samples_list, batch_size_list, 
            trainset, num_epoch)
    total_step = num_image // (batch_size_list[0] * num_epoch * world_size)

    batch_size = sum(batch_size_list)
    callback_logging = CallBackLogging(50, rank, total_step, batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, output_dir)

    global_step = 0 
    fp16 = opt.utils.fp16

    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    loss = AverageMeter()
    for epoch in range(0, num_epoch):
        total_step = trainer.train(epoch, global_step, train_loader, grad_amp, loss, callback_logging, callback_checkpoint) 
        global_step = total_step

def get_dataloader(local_rank, opt, seed):
    trainset = opt.dataset.trainset

    total_samples_list = []
    batch_size_list = []
    num_classes_list = []
    sampler_list = []
    dataset_list = []

    for dataname in trainset:
        rec_path = opt.dataset[dataname].rec_path
        idx_path = opt.dataset[dataname].idx_path
        num_samples = opt.dataset[dataname].num_samples
        num_classes = opt.dataset[dataname].num_classes
        batch_size = opt.dataset[dataname].batch_size
        train_set = MXFaceDataset(rec_path=rec_path, idx_path=idx_path, local_rank=local_rank, origin_preprocess=False)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, seed=seed)
        sampler_list.append(train_sampler)
        total_samples_list.append(num_samples)
        batch_size_list.append(batch_size)
        num_classes_list.append(num_classes)
        dataset_list.append(train_set)

    dataset = ConcatDataset(dataset_list)
    batch_sampler = DistributedMultiBatchSampler(samplers=sampler_list, 
            batch_size_list=batch_size_list, total_sample_num_list=total_samples_list)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=dataset,
        batch_sampler=batch_sampler, num_workers=opt.dataset.num_workers, pin_memory=True)
    return train_loader

class Trainer():

    def __init__(self, local_rank, world_size, num_classes_list, num_samples_list, 
            batch_size_list, dataset_name_list, num_epoch, emb_size=512, sample_rate=0.1):
        self.local_rank = local_rank
        self.world_size = world_size
        self.batch_size_list = batch_size_list

        self.total_batch_size = self.batch_size_list[0] * self.world_size
        self.num_classes_list = num_classes_list
        self.dataset_name_list = dataset_name_list
        self.emb_size = emb_size

        self.device = "cuda:{}".format(local_rank)
        self.backbone = None
        self.module_partial_fc = None
        self.module_partial_fc_list = []

        self.loss_fn = None
        self.num_image = num_samples_list[0]
        warmup_epoch = -1
        self.num_epoch = num_epoch
        self.fp16 = True
        self.warmup_step = self.num_image // self.total_batch_size * warmup_epoch
        self.total_step = self.num_image // self.total_batch_size * self.num_epoch
        self.decay_epoch = [8, 12, 15, 18]
        self.sample_rate = sample_rate
        self.opt_pfc_list = []
        self.scheduler_pfc_list = []
        self.batch_size = sum(self.batch_size_list)

        self.prepare()
        self.batch_size_offset = [0]

        batch_offset = 0
        for batch_size in self.batch_size_list:
            batch_offset += batch_size
            self.batch_size_offset.append(batch_offset)

    def network_init(self):
        self.backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)
        self.backbone.train()
        self.backbone.to(self.device)
        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        logging.info("init network at {} finished".format(self.local_rank))


    def set_loss(self):
        #self.loss_fn = get_loss("cosface")
        pass

    def set_tail(self, loss_fn):

        for dataname, num_classes, batch_size in zip(self.dataset_name_list, self.num_classes_list, 
                self.batch_size_list):
            prefix = "./{}".format(dataname)
            module_partial_fc = PartialFC(
                rank=self.local_rank, local_rank=self.local_rank, world_size=self.world_size, resume=False,
                batch_size=batch_size, margin_softmax=get_loss("cosface"), num_classes=num_classes,
                sample_rate=self.sample_rate, embedding_size=self.emb_size, prefix=prefix)
            self.module_partial_fc_list.append(module_partial_fc)

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

        for i in range(len(self.dataset_name_list)):
            opt_pfc = torch.optim.SGD(
                params=[{'params': self.module_partial_fc_list[i].parameters()}],
                lr=lr / 512 * self.batch_size_list[i] * self.world_size,
                momentum=0.9, weight_decay=5e-4)
            scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt_pfc, lr_lambda=lr_step_func)
            self.opt_pfc_list.append(opt_pfc)
            self.scheduler_pfc_list.append(scheduler_pfc)

    def prepare(self):
        self.network_init()
        self.set_loss()
        self.set_tail(loss_fn=self.loss_fn)
        self.set_optimizer(lr=0.1)

    def train(self, epoch, global_step, train_loader, grad_amp, loss,
            callback_logging, callback_checkpoint):
        for step, (img, label) in enumerate(train_loader):
            
            features = F.normalize(self.backbone(img))
            x_grad_list = []
            loss_v_list = []
            #dataset_features = []
            x_grad_list = []
            loss_v = 0
            for dataset_idx in range(len(self.dataset_name_list)):
                s = self.batch_size_offset[dataset_idx]
                e = self.batch_size_offset[dataset_idx + 1]
                feature_ = features[s:e, ]
                #dataset_features.append(feature_)
                label_ = label[s:e, ]
                x_grad, loss_v_ = self.module_partial_fc_list[dataset_idx].forward_backward(label_, feature_, self.opt_pfc_list[dataset_idx])
                x_grad_list.append(x_grad)
                loss_v += loss_v_

            if self.fp16:
                features.backward(grad_amp.scale(torch.cat(x_grad_list)))
                grad_amp.unscale_(self.opt_backbone)
                clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(self.opt_backbone)
                grad_amp.update()
            else:
                features.backward(torch.cat(x_grad_list))
                clip_grad_norm_(self.backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            for dataset_idx in range(len(self.dataset_name_list)):
                self.opt_pfc_list[dataset_idx].step()
                self.module_partial_fc_list[dataset_idx].update()
                self.opt_pfc_list[dataset_idx].zero_grad()

            self.opt_backbone.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(step + global_step, loss, epoch, self.fp16, self.scheduler_backbone.get_last_lr()[0], grad_amp)
            self.scheduler_backbone.step()

            for dataset_idx in range(len(self.dataset_name_list)):
                self.scheduler_pfc_list[dataset_idx].step()
        total_step = global_step + step
        callback_checkpoint(total_step, self.backbone, None)

        return total_step        

def main(args):
    config = args.config
    assert os.path.exists(config)
    opt = SourceFileLoader('module.name', './config/pretrain_config.py').load_module().opt
    size = opt.utils.num_gpu
    processes = []

    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, opt, run_train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--output_dir", type=str, default="/job_data/", help="")
    parser.add_argument("--config", type=str, default="./config/pretrain_config.py", help="")
    args = parser.parse_args()
    main(args)
