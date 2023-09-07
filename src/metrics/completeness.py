import torch
from torchmetrics import Metric
from sklearn.metrics import completeness_score


class Completeness(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("labels_true", default=[], dist_reduce_fx="cat")
        self.add_state("labels_pred", default=[], dist_reduce_fx="cat")
        self.metric_name = "completeness"

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Preds and targets should be torch tensors where each row corresponds to a single sample"""
        

        ## WORKS ONLY IF TENSOR IS 2D
        #Convert preds into sequnce of strings
        preds = list(map(lambda x: str("".join(map(str, x.detach().cpu().numpy()))),preds))
        ## WORKS ONLY IF TENSOR IS 2D
        #Convert targets into sequnce of strings
        targets = list(map(lambda x: str("".join(map(str, x.detach().cpu().numpy()))),targets))
        
        assert len(preds) == len(targets)
        
        self.labels_true.extend(targets)
        self.labels_pred.extend(preds)

    def compute(self):
        return completeness_score(labels_true=self.labels_true,
                                  labels_pred=self.labels_pred)
