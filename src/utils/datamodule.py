import torch
import random

from torch.utils.data import Sampler

class randomSupervisionSampler(Sampler):
    r"""Samples elements randomly. first a coin is tossed to see if it is a supervised batch or not.
    with replacement, so the user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, data_type_sampling_probability, generator=None, batch_size=32) -> None:
        self.data_source = data_source
        
        self.sup_len_xz = self.data_source.sup_len_xz
        self.sup_len_x = self.data_source.sup_len_x

        self.batch_size = batch_size
        self.stage = self.data_source.split

        self.data_type_sampling_probability = data_type_sampling_probability
        self._num_samples = len(self.data_source)
        self.generator = generator
        self.data_source_len = len(self.data_source)

        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
            self.generator = generator
        # else:
        #     generator = self.generator

        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples


    def __iter__(self):
        # torch.manual_seed(self.data_source.seed)
        # should I move it to the init?
        for _ in range(self.num_samples // self.batch_size):
            yield from self.sample_some_data(self.batch_size)

        yield from self.sample_some_data(self.num_samples % self.batch_size)

    def sample_some_data(self, num_samples):
        coin_toss = torch.rand(size=(1,), generator=self.generator).item()
        if coin_toss <= self.data_type_sampling_probability[0] and self.sup_len_xz>0:
            yield from torch.randint(high=self.sup_len_xz, size=(num_samples,), dtype=torch.int64, generator=self.generator).tolist()
        elif coin_toss <= (self.data_type_sampling_probability[0] + (1 - self.data_type_sampling_probability[0]) * (1 - self.data_type_sampling_probability[1])) and self.sup_len_x>0:
            yield from torch.randint(low=self.sup_len_xz, high=self.sup_len_xz + self.sup_len_x, size=(num_samples,), dtype=torch.int64, generator=self.generator).tolist()
        elif self.data_source_len > self.sup_len_xz + self.sup_len_x:
            yield from torch.randint(low=self.sup_len_xz + self.sup_len_x, high=self.data_source_len, size=(num_samples,), dtype=torch.int64, generator=self.generator).tolist()
        else:
            raise ValueError("No data to sample from")
        

    def __len__(self) -> int:
        return self.num_samples
    
    