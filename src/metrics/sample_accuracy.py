import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, pad_token=-1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64, device=self.device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.int64, device=self.device), dist_reduce_fx="sum")
        self.metric_name = "sample/acc"
        self.pad_token = pad_token

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Preds and targets should be torch tensors where each row corresponds to a single sample"""

        # # Convert input tensors to the device of the Accuracy instance
        # preds = preds.to(self.device)
        # targets = targets.to(self.device)

        assert preds.shape == targets.shape
        

        # remove padding from matches, if not at sentence level accuracy
        if len(targets.shape) == 1:
            matches = torch.eq(preds, targets)
            pred_pad_mask = torch.eq(preds, self.pad_token)
            target_pad_mask = torch.eq(targets, self.pad_token)
            matches = torch.logical_and(matches, torch.logical_not(target_pad_mask).reshape(matches.shape))
            num_correct = torch.sum(matches).long()
            num_total = torch.logical_not(target_pad_mask).sum().long()
        else:
            matches = torch.all(torch.eq(preds, targets), -1)
            num_correct = torch.sum(matches).long()
            num_total = torch.tensor(len(matches), device=self.device).long()

        self.total_correct += num_correct.to(self.device)
        self.total += num_total.to(self.device)

    def compute(self):
        if self.total_correct == 0 or self.total == 0:
            return torch.tensor(0.0, device=self.device)

        accuracy = self.total_correct.float() / self.total

        return accuracy
