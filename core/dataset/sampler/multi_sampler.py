import torch
from torch.utils.data import Sampler
import torch.distributed as dist

from typing import TypeVar, Optional, Iterator, List

class DistributedMultiBatchSampler(Sampler[list]):

    def __init__(self, samplers: list, batch_size_list: list, total_sample_num_list: list, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_sample_num_list = total_sample_num_list
        self.samplers = samplers
        self.batch_size_list = batch_size_list
        self.sample_iters = [sampler.__iter__() for sampler in self.samplers]
        self.epochs = [0] * len(self.samplers)
        self.batch_offset = [0]
        self.calcu_batch_offset()
        self.total_batch_size = sum(self.batch_size_list)
        self.step_per_epoch = (total_sample_num_list[0] // self.batch_size_list[0]) // self.num_replicas
        print("step per epoch: {}".format(self.step_per_epoch))

    def __iter__(self) -> Iterator[List[int]]:

        # subsample
        batch = []
        for _ in range(self.step_per_epoch):
            for sampler_idx, sampler in enumerate(self.samplers):
                cur_batch = []
                while True:
                    try:
                        idx = next(self.sample_iters[sampler_idx])
                    except StopIteration:
                        assert len(cur_batch) < self.batch_size_list[sampler_idx]
                        cur_epoch = self.epochs[sampler_idx]
                        cur_epoch += 1
                        sampler.set_epoch(cur_epoch)
                        self.epochs[sampler_idx] = cur_epoch

                        self.sample_iters[sampler_idx] = sampler.__iter__()
                        idx = next(self.sample_iters[sampler_idx])

                    cur_batch.append(idx + self.batch_offset[sampler_idx])
                    if len(cur_batch) == self.batch_size_list[sampler_idx]:
                        break

                batch.extend(cur_batch)
                if len(batch) == self.total_batch_size:
                    yield batch
                    batch = []
            #yield batch 

    def calcu_batch_offset(self):
        for sampler_idx, sampler in enumerate(self.samplers):
            data_len = self.total_sample_num_list[sampler_idx]
            cur_offset = self.batch_offset[sampler_idx]
            self.batch_offset.append(cur_offset + data_len)

    def __len__(self) -> int:
        return len(self.samplers[0]) // self.batch_size_list[0] // self.num_replicas

