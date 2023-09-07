import torch
from torchmetrics import Metric
from sklearn.metrics import homogeneity_score
import numpy as np


import torch
from torchmetrics import Metric
from sklearn.metrics import homogeneity_score


class SentenceHomogeneity(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("labels_true", default=[], dist_reduce_fx="cat")
        self.add_state("labels_pred", default=[], dist_reduce_fx="cat")
        self.metric_name = "homogeneity"

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
        score = homogeneity_score(labels_true=self.labels_true, labels_pred=self.labels_pred)
        return score
    
    


class TokenHomogeneity(Metric):

    def __init__(self, eos_token_id, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("labels_true", default=[], dist_reduce_fx="cat")
        self.add_state("labels_pred", default=[], dist_reduce_fx="cat")
        self.eos_token_id = eos_token_id
        self.metric_name = "token_homogeneity"

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Preds and targets should be torch tensors where each row corresponds to a single sample"""
        
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        preds_list, targets_list = [], []

        for i in range(preds.shape[0]):
            try:
                pred_idx = torch.where(preds[i, :]==self.eos_token_id)[0].ravel()[0]
            except:
                pred_idx = preds.shape[1]
            try:
                target_idx = torch.where(targets[i, :]==self.eos_token_id)[0].ravel()[0]
            except:
                target_idx = targets.shape[1]
            max_idx = max(pred_idx, target_idx)
            preds_list.extend(list(preds[i, :max_idx]))
            targets_list.extend(list(targets[i, :max_idx]))
        
        self.labels_true.extend(targets_list)
        self.labels_pred.extend(preds_list)

    def compute(self):
        return homogeneity_score(labels_true=self.labels_true,
                                 labels_pred=self.labels_pred)
    