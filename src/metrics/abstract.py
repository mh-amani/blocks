from abc import ABC
from typing import List, Set, Union, Tuple

import numpy as np
import torch
from torchmetrics import Metric


class AbstractTorchMetric(Metric, ABC):
    full_state_update = True

    def __init__(self, dist_sync_on_step=False):
        super().__init__(self, dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, correct: Union[int, torch.Tensor], predicted: Union[int, torch.Tensor], target: Union[int, torch.Tensor]):
        return NotImplementedError()

    def compute(self):
        return NotImplementedError()

    
        