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

    def __init__(self, data_source, p_sup, generator=None, batch_size=32) -> None:
        self.data_source = data_source
        self.sup_len = self.data_source.sup_len
        self.batch_size = batch_size
        self.stage = self.data_source.split
        self.p_sup = p_sup
        self._num_samples = len(self.data_source)
        self.generator = generator
        


    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for _ in range(self.num_samples // self.batch_size):
            if random.random() <= self.p_sup and self.sup_len:
                yield from torch.randint(high=self.sup_len, size=(self.batch_size,), dtype=torch.int64, generator=generator).tolist()
            else:
                yield from torch.randint(low=self.sup_len, high=n, size=(self.batch_size,), dtype=torch.int64, generator=generator).tolist()
        if random.random() <= self.p_sup and self.sup_len:
            yield from torch.randint(high=self.sup_len, size=(self.num_samples % self.batch_size,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randint(low=self.sup_len, high=n, size=(self.batch_size % self.batch_size,), dtype=torch.int64, generator=generator).tolist()
        

    def __len__(self) -> int:
        return self.num_samples
    
    