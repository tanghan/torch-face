import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse

var_target = []

dataset_dict = {"lfw": "/home/users/han.tang/data/public_face_data/glint/glint360k/lfw.bin"}

from core.models.resnet import iresnet

class Eval(object):

    def __init__(self, local_rank, fp16=True, emb_size=512):
        self.backbone = None
        self.local_rank = local_rank
        self.fp16 = fp16
        self.emb_size = emb_size
        self.device = "cuda:{}".format(local_rank)

    def network_init(self, weight_path):
        self.backbone = iresnet.iresnet100(dropout=0.0, fp16=self.fp16, num_features=self.emb_size)
        self.backbone.to(self.device)

        self.backbone.load_state_dict()


        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        logging.info("init network at {} finished".format(self.local_rank))
        self.backbone.eval()

    def prepare(self):
        self.network_init()

    @torch.no_grad()
    def eval(self, dataloader):
        data = data_list[i]
        embeddings = None
        ba = 0
        embeddings_list = []
        for step, data in enumerate(dataloader):
            imgs, flip_imgs = data
            out = self.backbone(imgs)
            if step % 10000 == 0:
                print("process {}".format(step)
            '''
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
            '''
            embeddings_list.append(embeddings)


def build_dataset(bin_path, local_rank, batch_size):
    dataset = MXBinFaceDataset(bin_path, local_rank)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = EvalDataLoader(
        local_rank=local_rank, dataset=dataset, batch_size=batch_size,
        sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
    return dataloader

def init_process(rank, world_size, args, fn):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group(backend='nccl',
		rank=rank, world_size=world_size)
    fn(args, rank, world_size)

def run(args, rank, world_size):
    dataset_name = args.dataset
    bin_path = dataset_dict[dataset_name]
    test = Eval(local_rank, emb_size=512, fp16=True)
    dataloader = build_dataset(bin_path, rank, batch_size=64)

@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def main(args):
    gpu_num = args.gpu_num
    size = gpu_num
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--dataset", type=str, default="lfw", help="")
    parser.add_argument("--gpu_num", type=int, default=2, help="")
    args = parser.parse_args()
    main(args)

