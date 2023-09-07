from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractDiscreteLayer(nn.Module):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__()      

    @abstractmethod
    def linear_layer(self, in_features, out_features):
        pass

    @abstractmethod
    def loss(label_ids, preds):
        pass
        
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def discretize(self, x) -> dict:
        pass

    @abstractmethod
    def embed_from_id(self, x):
        pass

