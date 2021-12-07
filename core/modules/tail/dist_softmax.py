import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter
from core.modules.loss.MagFace import MagFace

from torch.autograd import Function

class GatherFeature2_(Function):

    @staticmethod
    def forward(ctx, features):

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size, embedding_size = features.size()
        total_features = torch.zeros(
            size=[batch_size * world_size, embedding_size], device=features.device)
        dist.all_gather(list(total_features.chunk(world_size, dim=0)), features.data)

        #total_features.zero_()
        #total_features[rank * batch_size:(rank + 1) * batch_size, ] = features.clone().detach()
        dist.barrier()
        print("total features shape: {}, batch_size: {}".format(total_features.size(), batch_size))

        return total_features

    @staticmethod
    def backward(ctx, grad_output):

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size = grad_output.size()[0] // world_size
        #if rank == 0:
        #    print(grad_output)
        #grad_output.mul_(world_size)
        #grad_output.div_(world_size)
        print("grad output shape: {}, batch_size: {}, rank: {}".format(grad_output.size(), batch_size, rank))
        #if rank == 0:
        #    print(grad_output)
        if rank == 0:
            print("grad output: {}".format(grad_output[rank * batch_size:(rank + 1) * batch_size, :10]))

        return grad_output[rank * batch_size:(rank + 1) * batch_size, :]

class GatherFeature1_(Function):

    @staticmethod
    def forward(ctx, features, total_features):

        rank = dist.get_rank()
        batch_size = features.size()[0]
        #total_features.zero_()
        #total_features[rank * batch_size:(rank + 1) * batch_size, ] = features.clone().detach()
        dist.all_reduce(total_features, dist.ReduceOp.SUM)
        dist.barrier()
        print("total features shape: {}, batch_size: {}".format(total_features.size(), batch_size))

        return total_features

    @staticmethod
    def backward(ctx, grad_output):

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size = grad_output.size()[0] // world_size
        #if rank == 0:
        #    print(grad_output)
        #grad_output.mul_(world_size)
        #grad_output.div_(world_size)
        print("grad output shape: {}, batch_size: {}, rank: {}".format(grad_output.size(), batch_size, rank))
        #if rank == 0:
        #    print(grad_output)
        if rank == 0:
            print("grad output: {}".format(grad_output[rank * batch_size:(rank + 1) * batch_size, :10]))

        return grad_output[rank * batch_size:(rank + 1) * batch_size, :], None


class Matmul_(Function):

    @staticmethod
    def forward(ctx, features, w, total_features):
        dist.all_reduce(total_features, dist.ReduceOp.SUM)
        dist.barrier()
        outputs = linear(total_features, w)

        ctx.save_for_backward(total_features, w)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        total_features, w = ctx.saved_tensors
        batch_size = total_features.size()[0] // world_size
        grad_logits = torch.mm(grad_output, w)
        dist.all_reduce(grad_logits, dist.ReduceOp.SUM)
        dist.barrier()

        grad_w = torch.mm(torch.t(total_features), grad_output)
        if rank == 0:
            print("mat mul grad: {}".format(grad_logits[rank * batch_size:(rank + 1) * batch_size, :10] * world_size))

        return grad_logits[rank * batch_size:(rank + 1) * batch_size, ] * world_size, torch.t(grad_w), None

class Matmul2_(Function):

    @staticmethod
    def forward(ctx, features, w):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        batch_size, embedding_size = features.size()
        total_features = torch.zeros(
            size=[batch_size * world_size, embedding_size], device=features.device)
        dist.all_gather(list(total_features.chunk(world_size, dim=0)), features.data)

        #total_features.zero_()
        #total_features[rank * batch_size:(rank + 1) * batch_size, ] = features.clone().detach()
        dist.barrier()

        dist.barrier()
        outputs = linear(total_features, w)

        ctx.save_for_backward(total_features, w)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        total_features, w = ctx.saved_tensors
        batch_size = total_features.size()[0] // world_size
        grad_logits = torch.mm(grad_output, w)
        dist.all_reduce(grad_logits, dist.ReduceOp.SUM)
        dist.barrier()

        grad_w = torch.mm(torch.t(total_features), grad_output)
        if rank == 0:
            print("mat mul grad: {}".format(grad_logits[rank * batch_size:(rank + 1) * batch_size, :10] * world_size))

        return grad_logits[rank * batch_size:(rank + 1) * batch_size, ] * world_size, torch.t(grad_w)

class Matmul3_(Function):

    @staticmethod
    def forward(ctx, features, w):

        outputs = linear(features, w)
        ctx.save_for_backward(features, w)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        total_features, w = ctx.saved_tensors
        batch_size = total_features.size()[0] // world_size
        grad_logits = torch.mm(grad_output, w)
        dist.all_reduce(grad_logits, dist.ReduceOp.SUM)
        dist.barrier()

        grad_w = torch.mm(torch.t(total_features), grad_output)

        return grad_logits * world_size, torch.t(grad_w)



class SoftmaxFunc_(Function):

    @staticmethod
    def forward(ctx, logits: torch.Tensor, total_labels: torch.Tensor):
        max_fc = torch.max(logits, dim=1, keepdim=True)[0]
        dist.all_reduce(max_fc, dist.ReduceOp.MAX)
        dist.barrier()
        logits_exp = torch.exp(logits - max_fc)
        logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
        dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)
        dist.barrier()

        logits_exp.div_(logits_sum_exp)
        grad = logits_exp * 1.
        ctx.save_for_backward(grad, total_labels)
        index = torch.where(total_labels != -1)[0]
        loss = logits_exp[index].gather(1, total_labels[index].view(-1, 1))
        loss = loss.clamp_min_(1e-30).log_().sum() * (-1.)
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        loss = loss / (total_labels.size()[0])

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad, total_labels = ctx.saved_tensors
        rank = dist.get_rank()

        index = torch.where(total_labels != -1)[0]
        one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
        one_hot.scatter_(1, total_labels[index, None], 1)
        grad[index] -= one_hot
        grad.div_(total_labels.size()[0])

        return grad * grad_output, None


class DistSoftmax(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, embedding_size=512, prefix="./", seed=1234):
        """
        rank: int
            Unique process(GPU) ID from 0 to world_size - 1.
        local_rank: int
            Unique process(GPU) ID within the server from 0 to 7.
        world_size: int
            Number of GPU.
        batch_size: int
            Batch size on current rank(GPU).
        resume: bool
            Select whether to restore the weight of softmax.
        margin_softmax: callable
            A function of margin softmax, eg: cosface, arcface.
        num_classes: int
            The number of class center storage in current rank(CPU/GPU), usually is total_classes // world_size,
            required.
        embedding_size: int
            The feature dimension, default is 512.
        prefix: str
            Path for save checkpoint, default is './'.
        """
        super(DistSoftmax, self).__init__()
        #
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.num_classes: int = num_classes
        self.embedding_size: int = embedding_size

        self.device: torch.device = torch.device(self.local_rank)
        self.generator = None
        if resume is not True:
            generator = torch.Generator(device=self.device)
            self.generator = generator.manual_seed(seed + rank)
        self.margin_softmax: callable = margin_softmax
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)

        self.weight_name = os.path.join(self.prefix, "rank_{}_softmax_weight.pt".format(self.rank))
        self.total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device, requires_grad=False)

        if resume:
            try:
                logging.info("softmax weight resume from {}".format(self.weight_name))
                weight: torch.Tensor = torch.load(self.weight_name).cuda(local_rank)

                if weight.shape[0] != self.num_local:
                    logging.info("shape not equal, {} != {} or {} != {}".format(weight.shape[0], self.num_local, self.num_local))
                    raise IndexError
                logging.info("softmax weight resume successfully!")
            except (FileNotFoundError, KeyError, IndexError):
                weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device, generator=self.generator)
                logging.info("softmax weight init!")
        else:
            weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device, generator=self.generator)
            #weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            logging.info("softmax weight init successfully!")
        #save_path = "dist_softmax_init_{}_w.pt".format(rank)
        #torch.save(weight.cpu(), save_path)
        self.weight = torch.nn.Parameter(weight)
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)
        self.iter = 0

    def save_params(self):
        """ Save softmax weight for each rank on prefix
        """
        torch.save(self.weight.data, self.weight_name)

    @torch.no_grad()
    def sample(self, total_label):
        """
        Sample all positive class centers in each rank, and random select neg class centers to filling a fixed
        `num_sample`.

        total_label: tensor
            Label after all gather, which cross all GPUs.
        """
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        
    def forward(self, features, label):
        """ Partial fc forward, `logits = X * sample(W)`
        """
        total_label, norm_weight = self.prepare(label)

        with torch.no_grad():
            self.total_features.zero_()
            self.total_features[self.rank * self.batch_size:(self.rank + 1) * self.batch_size, ] = features.clone().detach()

        torch.cuda.current_stream().wait_stream(self.stream)
        dist.barrier()

        gathered_features = GatherFeature1_.apply(features, self.total_features)
        norm_feat = F.normalize(gathered_features)
        #logits = F.linear(gathered_features, norm_weight)
        #logits = Matmul_.apply(features, norm_weight, self.total_features)
        #logits = Matmul2_.apply(features, norm_weight)
        #gathered_features = GatherFeature2_.apply(features)
        logits = Matmul3_.apply(norm_feat, norm_weight)

        loss_g = None
        if isinstance(self.margin_softmax, MagFace):
            logits, loss_g = self.margin_softmax(logits, gathered_features, total_label)
        else:
            logits = self.margin_softmax(logits, total_label)

        save_path = "dist_logits_{}_{}.pt".format(self.iter, self.rank)

        torch.save(logits.cpu(), save_path)
        loss = SoftmaxFunc_.apply(logits, total_label)

        self.iter += 1

        return loss, loss_g

    def prepare(self, label):
        """
        get sampled class centers for cal softmax.

        label: tensor
            Label tensor on each rank.
        optimizer: opt
            Optimizer for partial fc, which need to get weight mom.
        """
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            dist.barrier()
            self.sample(total_label)
            norm_weight = normalize(self.weight)
            return total_label, norm_weight
