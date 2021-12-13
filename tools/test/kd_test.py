import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse

import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn
import math
from scipy.stats import norm

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
#from core.modules.tail.partial_fc import PartialFC
from core.modules.tail.dist_partial_fc import PartialFC
from core.modules.tail.dist_softmax import DistSoftmax 
from core.models.resnet import iresnet
from core.modules.loss.ArcFace import ArcFace
from core.modules.local_loss.ArcFace import ArcFace as localArcFace
from core.modules.local_loss.CurricularFace import CurricularFace as localCurricularFace
from core.modules.local_loss.CircleLoss import CircleLoss as localCircleLoss
from core.modules.local_loss.MagFace import MagFace as localMagFace
from core.modules.loss.CircleLoss import CircleLoss
from torch.cuda.amp import GradScaler
from utils.utils_amp import MaxClipGradScaler
from core.modules.loss.head_def import HeadFactory

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

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

    def __init__(self, local_rank, world_size, s_emb_size, t_emb_size, fp16=True):
        super(StudentBackbone, self).__init__()
        self.out = None
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

    def build_margin(self, teacher_bns):

        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())


    def build_backbone(self):
        self.s_backbone = iresnet.iresnet18(dropout=0.0, fp16=self.fp16, num_features=self.s_emb_size)


    def forward(self, x, t_feats):
        feats, logits = self.s_backbone.extract_feature(x)
        s_feats = []
        loss_distill = 0.
        for i in range(self.feat_num):
            s_feats.append(self.Connectors[i](feats[i]))
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (self.feat_num - i - 1)

        return loss_distill, logits


def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run_test(args, rank, world_size, fp16=False):
    torch.cuda.set_device(rank)
    batch_size = args.batch_size
    #loss_type = args.loss_type
    num_classes = 12
    #num_samples = 16
    num_samples = 48
    input_shape = (3, 112, 112)
    s_emb_size = 256
    t_emb_size = 512

    rng = np.random.RandomState(1234)

    dataloaer = build_dataset(rng, batch_size, num_classes, num_samples, input_shape)
    #head_factory = HeadFactory(rank, world_size, "MagFace", "./config/head_conf.yaml")
    #head_factory = HeadFactory(rank, world_size, "CircleLoss", "./config/head_conf.yaml")
    head_factory = HeadFactory(rank, world_size, "CurricularFace", "./config/head_conf.yaml")
    margin_softmax = head_factory.get_head()
    #margin_softmax = CircleLoss(rank, world_size)
    t_backbone, s_backbone, fc = build_model(rank, world_size, num_classes, 
            batch_size, margin_softmax, t_emb_size=t_emb_size, s_emb_size=s_emb_size, fp16=fp16)
    opt_backbone = torch.optim.SGD(
            params=[{'params': s_backbone.parameters()}],
            lr=0.1,
            momentum=0.9, weight_decay=5e-4)
    opt_pfc = torch.optim.SGD(
        params=[{'params': fc.parameters()}],
        lr=0.1,
        momentum=0.9, weight_decay=5e-4)

    loss_fn = torch.nn.CrossEntropyLoss()
    #scaler = GradScaler()
    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    feat_num = 4
    for data_idx, data in enumerate(dataloaer):
        img, label = data
        img = img.cuda(rank)
        label = label.to("cuda:{}".format(rank))
        with torch.no_grad():
            t_feats, _ = t_backbone.extract_feature(img)
        distill_loss, s_logits = s_backbone(img, t_feats)
        s_features = F.normalize(s_logits)



def build_dataset(rng, batch_size, num_classes, num_samples, input_shape):
    input_feat = rng.normal(size=(num_samples, input_shape[0], input_shape[1],input_shape[2]))
    input_feat = torch.tensor(input_feat, dtype=torch.float32)
    labels = rng.randint(low=0, high=num_classes, size=(num_samples, ))
    print("labels: ", labels)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_feat, labels)
    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return dataloader


def build_model(local_rank, world_size, num_classes, batch_size, margin_softmax, t_emb_size=512, s_emb_size=256, fp16=True):

    t_backbone = iresnet.iresnet18(dropout=0.0, fp16=fp16, num_features=t_emb_size)

    t_backbone = t_backbone.cuda(local_rank)
    t_backbone.eval()

    s_backbone = StudentBackbone(local_rank, world_size, t_emb_size=t_emb_size, s_emb_size=s_emb_size, fp16=fp16)
    s_backbone.build_margin(t_backbone.get_bn_before_relu())
    s_backbone = s_backbone.cuda(local_rank)
    s_backbone = torch.nn.parallel.DistributedDataParallel(
        module=s_backbone, broadcast_buffers=False, device_ids=[local_rank])
    s_backbone.train()

    fc = DistSoftmax(
        rank=local_rank, local_rank=local_rank, world_size=world_size, 
        batch_size=batch_size, resume=False, margin_softmax=margin_softmax, num_classes=num_classes,
        embedding_size=s_emb_size)

    #fc.cuda(local_rank)
    return t_backbone, s_backbone, fc

def main(args):
    torch.manual_seed(1234)
    size = args.gpu_num

    processes = []
    #torch.use_deterministic_algorithms(True)
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, run_test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test loss")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    parser.add_argument("--batch_size", type=int, default=4, help="")
    args = parser.parse_args()
    main(args)

