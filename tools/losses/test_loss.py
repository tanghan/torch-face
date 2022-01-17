import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse

import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from core.modules.tail.partial_fc import PartialFC
from core.modules.tail.dist_partial_fc import PartialFC as DistPartialFC
from core.modules.tail.dist_softmax import DistSoftmax 
from core.modules.tail.dist_partial_softmax import DistPartialSoftmax 
#from core.models.resnet import iresnet_test as iresnet
from core.models.resnet import iresnet as iresnet
from core.modules.loss.ArcFace import ArcFace
from core.modules.local_loss.ArcFace import ArcFace as localArcFace
from core.modules.local_loss.CurricularFace import CurricularFace as localCurricularFace
from core.modules.local_loss.CircleLoss import CircleLoss as localCircleLoss
from core.modules.local_loss.MagFace import MagFace as localMagFace
from core.modules.loss.CircleLoss import CircleLoss
from torch.cuda.amp import GradScaler
from utils.utils_amp import MaxClipGradScaler
from core.modules.loss.head_def import HeadFactory

def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run_test(args, rank, world_size, fp16=False):
    torch.cuda.set_device(rank)
    batch_size = args.batch_size
    loss_type = args.loss_type
    num_classes = 100
    compare_idx = args.compare_idx
    #num_samples = 16
    num_samples = 48
    input_shape = (3, 112, 112)
    sample_rate = args.sample_rate

    partial = args.partial
    rng = np.random.RandomState(1234)
    if partial == 0:
        batch_size = args.gpu_num * batch_size
        world_size = args.gpu_num
    print("start with partial {}, use fp16: {}".format(partial, fp16))

    dataloaer = build_dataset(rng, batch_size, num_classes, num_samples, input_shape)
    #head_factory = HeadFactory(rank, world_size, "MagFace", "./config/head_conf.yaml")
    head_factory = HeadFactory(rank, world_size, "CircleLoss", "./config/head_conf.yaml")
    #head_factory = HeadFactory(rank, world_size, "CurricularFace", "./config/head_conf.yaml")
    margin_softmax = head_factory.get_head()
    #margin_softmax = CircleLoss(rank, world_size)
    backbone, fc = build_model(rank, world_size, num_classes, batch_size, margin_softmax, fp16=fp16, partial=partial, sample_rate=sample_rate)
    opt_backbone = torch.optim.SGD(
            params=[{'params': backbone.parameters()}],
            lr=0.1,
            momentum=0.9, weight_decay=5e-4)
    opt_pfc = torch.optim.SGD(
        params=[{'params': fc.parameters()}],
        lr=0.1,
        momentum=0.9, weight_decay=5e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    '''
    print("----- before train -------")
    for p in backbone.parameters():
        print(p)

    print("----- start train -------")
    '''
    #scaler = GradScaler()
    grad_amp = MaxClipGradScaler(batch_size, 128 * batch_size, growth_interval=100) if fp16 else None
    for data_idx, data in enumerate(dataloaer):
        img, label = data
        img = img.cuda(rank)
        #print("img: ", img[0, 0, 0, :])
        label = label.to("cuda:{}".format(rank))
        features = backbone(img)
        print("last fea: ", features[0, :5])
        #features = F.normalize(backbone(img))
        features.retain_grad()

        if partial == 0:
            #x_grad, loss_v = fc.forward_backward(label, features, opt_pfc)
            
            feature_path = "p{}_fea_{}.pt".format(compare_idx, data_idx)
            torch.save(features.cpu(), feature_path)
            logits = fc(features, label)
            #logits, loss_g = fc(features, label)
            loss_v = loss_fn(logits, label)
            #loss_v = loss_v + torch.mean(loss_g)

            logits_path = "p{]_logits_{}.pt".format(compare_idx, data_idx)
            torch.save(logits.cpu(), logits_path) 
        elif partial == 1:

            feature_path = "p{}_fea_{}_{}.pt".format(compare_idx, data_idx, rank)
            torch.save(features.cpu(), feature_path)
            loss_v, loss_g = fc(features, label)

            if loss_g is not None:
                #print("loss g shape: {}".format(loss_g.size()))
                loss_v = loss_v + torch.mean(loss_g) / world_size

        elif partial == 2:

            feature_path = "p{}_fea_{}_{}.pt".format(compare_idx, data_idx, rank)
            torch.save(features.cpu(), feature_path)
            norm_features = F.normalize(features)
            x_grad, loss_v = fc.forward_backward(label, norm_features, opt_pfc)

        elif partial == 3:
            feature_path = "p{}_fea_{}_{}.pt".format(compare_idx, data_idx, rank)
            torch.save(features.cpu(), feature_path)

            x_grad, loss_v = fc.forward_backward(label, features, opt_pfc)
        elif partial == 4:
            feature_path = "p{}_fea_{}_{}.pt".format(compare_idx, data_idx, rank)
            torch.save(features.cpu(), feature_path)

            loss_v, loss_g = fc.forward(features, label, opt_pfc)

        print("rank: {}, loss shape: {}, loss: {}".format(rank, loss_v.shape, loss_v))
        if fp16:
            if partial == 2 or partial == 3:
                features.backward(grad_amp.scale(x_grad))
                #scaler.scale(loss_v).backward()
            else:
                grad_amp.scale(loss_v).backward()
            grad_amp.unscale_(opt_backbone)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2, error_if_nonfinite=True)
            grad_amp.step(opt_backbone)
            grad_amp.update()
        else:
            if partial == 2:
                norm_features.backward(x_grad)
            elif partial == 3:
                features.backward(x_grad)
            else:
                loss_v.backward()
            #torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()

        if partial == 0:
            fea_grad_path = "p{}_fea_grad_{}.pt".format(compare_idx, data_idx)
        else:
            fea_grad_path = "p{}_fea_grad_{}_{}.pt".format(compare_idx, data_idx, rank)
        torch.save(features.grad.cpu(), fea_grad_path)

        opt_pfc.step()
        if partial == 2 or partial == 3 or partial == 4:
            fc.update()
        opt_backbone.zero_grad()
        opt_pfc.zero_grad()


        #print("fc grad:", fc.weight.grad)
        max_display = 0
        display_iter = 0
        if partial == 0:
            for p in fc.module.parameters():
                torch.save(p.cpu(), "p{}_fc_{}.pt".format(compare_idx, data_idx))
                #print(p[:, :5])

            for p in backbone.module.parameters():
                if display_iter > max_display:
                    break
                torch.save(p.cpu(), "p{}_backbone_{}.pt".format(compare_idx, data_idx))
                #print("backbone p value: {} ".format(p[0, :2, :, :]))
                display_iter += 1
        elif partial == 1:
            for p in fc.parameters():
                torch.save(p.cpu(), "p{}_fc_{}_{}.pt".format(compare_idx, data_idx, rank))
                pass
                #print(p[:, :5])

            for p in backbone.module.parameters():
                if display_iter > max_display:
                    break
                if rank == 0:
                    torch.save(p.cpu(), "p{}_backbone_{}.pt".format(compare_idx, data_idx))
                #print("backbone p value: {} ".format(p[0, :2, :, :]))
                display_iter += 1
        elif partial == 2 or partial == 3 or partial == 4:
            for p in fc.parameters():
                torch.save(p.cpu(), "p{}_fc_{}_{}.pt".format(compare_idx, data_idx, rank))
                pass
                #print(p[:, :5])

            for p in backbone.module.parameters():
                if display_iter > max_display:
                    break
                if rank == 0:
                    torch.save(p.cpu(), "p{}_backbone_{}.pt".format(compare_idx, data_idx))
                #print("backbone p value: {} ".format(p[0, :2, :, :]))
                display_iter += 1




def build_dataset(rng, batch_size, num_classes, num_samples, input_shape):
    input_feat = rng.normal(0, 1, size=(num_samples, input_shape[0], input_shape[1],input_shape[2]))
    input_feat = torch.tensor(input_feat, dtype=torch.float32)
    labels = rng.randint(low=0, high=num_classes, size=(num_samples, ))
    print("labels: ", labels)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_feat, labels)
    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    return dataloader


def build_model(local_rank, world_size, num_classes, batch_size, margin_softmax, embedding_size=512, fp16=True, partial=False, sample_rate=1.):

    backbone = iresnet.iresnet18(dropout=0.0, fp16=fp16, num_features=embedding_size)
    backbone = backbone.cuda(local_rank)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    if partial == 0: 
        #fc = localMagFace(local_rank, world_size, feat_dim=embedding_size, num_class=num_classes, scale=64, margin_am=0.0, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=20)
        #fc = localArcFace(local_rank, world_size, feat_dim=embedding_size, num_class=num_classes)
        fc = localCircleLoss(local_rank, world_size, feat_dim=embedding_size, num_class=num_classes)
        #fc = localCurricularFace(local_rank, world_size, feat_dim=embedding_size, num_class=num_classes)
        fc = fc.cuda(local_rank)
        fc = torch.nn.parallel.DistributedDataParallel(
            module=fc, broadcast_buffers=False, device_ids=[local_rank])

    elif partial == 1:

        fc = DistSoftmax(
            rank=local_rank, local_rank=local_rank, world_size=world_size, 
            batch_size=batch_size, resume=False, margin_softmax=margin_softmax, num_classes=num_classes,
            embedding_size=embedding_size)
    elif partial == 2:
        fc = PartialFC(rank=local_rank, local_rank=local_rank, world_size=world_size, sample_rate=sample_rate,
            batch_size=batch_size, resume=False, margin_softmax=margin_softmax, num_classes=num_classes,
            embedding_size=embedding_size)
    elif partial == 3:
        fc = DistPartialFC(rank=local_rank, local_rank=local_rank, world_size=world_size, sample_rate=sample_rate,
            batch_size=batch_size, resume=False, margin_softmax=margin_softmax, num_classes=num_classes,
            embedding_size=embedding_size)

    elif partial == 4:
        fc = DistPartialSoftmax(rank=local_rank, local_rank=local_rank, world_size=world_size, sample_rate=sample_rate,
            batch_size=batch_size, resume=False, margin_softmax=margin_softmax, num_classes=num_classes,
            embedding_size=embedding_size)

    else:
        raise AssertionError("error partial type: {}".format(partial))

    #fc.cuda(local_rank)
    return backbone, fc

def main(args):
    torch.manual_seed(1234)
    size = args.gpu_num
    partial = args.partial
    if partial == 0:
        size = 1

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
    parser.add_argument("--compare_idx", type=int, default=0, help="")
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--loss_type", type=str, default="arcface", help="")
    parser.add_argument("--partial", type=int, default=0, help="")
    parser.add_argument("--sample_rate", type=float, default=1., help="")
    args = parser.parse_args()
    main(args)

