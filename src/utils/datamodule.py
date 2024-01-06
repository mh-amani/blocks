import torch
import random
from typing import Optional, Iterator, List

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

import math


class randomSupervisionSampler(Sampler):
    r"""Samples elements randomly. first a coin is tossed to see if it is a supervised batch or not.
    with replacement, so the user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, data_type_sampling_probability, seed=42, batch_size=32, generator=None) -> None:
        self.data_source = data_source
        
        self.sup_len_xz = self.data_source.sup_len_xz
        self.sup_len_unsup = self.data_source.sup_len_unsup

        self.batch_size = batch_size
        self.data_type_sampling_probability = data_type_sampling_probability
    
        self.data_source_len = len(data_source)
        self.batch_size = batch_size

        self._num_samples = self.data_source_len 
        self.num_batch_iters = (self._num_samples // self.batch_size) + (self._num_samples % self.batch_size > 0)
        self.total_size = self.batch_size * self.num_batch_iters
        self.epoch = 0
        self.data_type_sampling_probability = data_type_sampling_probability
        self.seed = seed

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        # self.generator = generator
        self.state_list = []
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return self._num_samples


    def __iter__(self) -> Iterator[List[int]]:
        # DistributedSampler pytorch implemention __iter__
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        
        # g = self.generator

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # print(g.get_state())
        # self.state_list.append(g.get_state())
        # check if elements of the seed are the same or not
        # for i in range(0, len(self.state_list)):
            # print(torch.all(self.state_list[-1] == self.state_list[i]))


        # print(g.seed())

        coin_tosses = torch.rand(size=(self.num_batch_iters, ), generator=g).tolist()
        indices = []
        
        for coin_toss in coin_tosses:
            if coin_toss <= self.data_type_sampling_probability[0] and self.sup_len_xz>0:
                indices.extend(torch.randint(high=self.sup_len_xz, size=(self.batch_size,), dtype=torch.int64, generator=g).tolist())
            elif coin_toss <= (self.data_type_sampling_probability[0] + (1 - self.data_type_sampling_probability[0]) * (1 - self.data_type_sampling_probability[1])) and self.sup_len_unsup>0:
                indices.extend(torch.randint(low=self.sup_len_xz, high=self.sup_len_xz + self.sup_len_unsup, size=(self.batch_size,), dtype=torch.int64, generator=g).tolist())
            elif self.data_source_len > self.sup_len_xz + self.sup_len_unsup:
                indices.extend(torch.randint(low=self.sup_len_xz + self.sup_len_unsup, high=self.data_source_len, size=(self.batch_size, ), dtype=torch.int64, generator=g).tolist())
            else:
                raise ValueError("No data to sample from")

        # print(len(set(indices)))
        # print('id min', min(indices), 'id max', max(indices))

        assert len(indices) == self.total_size
        
        # print(len(indices), self.total_size, self.rank, self.num_replicas)
        # assert len(indices) == self.num_samples
        
        return iter(indices)


    # def __iter__(self):
    #     # torch.manual_seed(self.data_source.seed)
    #     # should I move it to the init?
    #     # g = torch.Generator()
    #     # g.manual_seed(self.seed + self.epoch)

    #     g = self.generator
    #     for _ in range(self.num_batch_iters):
    #         yield from self.sample_some_data(self.batch_size, g)

    #     # yield from self.sample_some_data(self._num_samples % self.batch_size, g)

    # def sample_some_data(self, size, g):
    #     coin_toss = torch.rand(size=(1,), generator=g).item()
    #     if coin_toss <= self.data_type_sampling_probability[0] and self.sup_len_xz>0:
    #         yield from torch.randint(high=self.sup_len_xz, size=(size,), dtype=torch.int64, generator=g).tolist()
    #     elif coin_toss <= (self.data_type_sampling_probability[0] + (1 - self.data_type_sampling_probability[0]) * (1 - self.data_type_sampling_probability[1])) and self.sup_len_x>0:
    #         yield from torch.randint(low=self.sup_len_xz, high=self.sup_len_xz + self.sup_len_x, size=(size,), dtype=torch.int64, generator=g).tolist()
    #     elif self.data_source_len > self.sup_len_xz + self.sup_len_x:
    #         yield from torch.randint(low=self.sup_len_xz + self.sup_len_x, high=self.data_source_len, size=(size,), dtype=torch.int64, generator=g).tolist()
    #     else:
    #         raise ValueError("No data to sample from")
        

    def __len__(self) -> int:
        return self._num_samples
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    

class randomSupervisionSamplerDDP(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, **kwargs) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        # Initialize your stuff
        self.data_type_sampling_probability = kwargs['data_type_sampling_probability']
        
        self.sup_len_xz = dataset.sup_len_xz
        self.sup_len_unsup = dataset.sup_len_unsup
        self.data_source_len = len(dataset)
        self.batch_size = kwargs['batch_size']
        
        self.replica_size = self.batch_size * self.num_replicas
        self.num_batch_iter = math.ceil(self.data_source_len / self.replica_size)
        self.total_size = self.num_batch_iter * self.replica_size
        # print('num_samples', self.num_samples, 'total_size', self.total_size, 'replica_size', self.replica_size, 'num_batch_iter', self.num_batch_iter)


    def __iter__(self) -> Iterator[List[int]]:
        # DistributedSampler pytorch implemention __iter__
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        # if self.shuffle:
        #     # deterministically shuffle based on epoch and seed
        #     g = torch.Generator()
        #     g.manual_seed(self.seed + self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        # else:
        #     indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        coin_tosses = torch.rand(size=(self.num_batch_iter, ), generator=g).tolist()
        indices = []
        
        for coin_toss in coin_tosses:
            if coin_toss <= self.data_type_sampling_probability[0] and self.sup_len_xz>0:
                indices.extend(torch.randint(high=self.sup_len_xz, size=(self.replica_size,), dtype=torch.int64, generator=g).tolist())
            elif coin_toss <= (self.data_type_sampling_probability[0] + (1 - self.data_type_sampling_probability[0]) * (1 - self.data_type_sampling_probability[1])) and self.sup_len_unsup>0:
                indices.extend(torch.randint(low=self.sup_len_xz, high=self.sup_len_xz + self.sup_len_unsup, size=(self.replica_size,), dtype=torch.int64, generator=g).tolist())
            elif self.data_source_len > self.sup_len_xz + self.sup_len_unsup:
                indices.extend(torch.randint(low=self.sup_len_xz + self.sup_len_unsup, high=self.data_source_len, size=(self.replica_size, ), dtype=torch.int64, generator=g).tolist())
            else:
                raise ValueError("No data to sample from")


        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        # code.interact(local=locals())

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        # print(len(indices), self.total_size, self.rank, self.num_replicas)
        # assert len(indices) == self.num_samples
        
        return iter(indices)


    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
