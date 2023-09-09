from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractDiscreteLayer(nn.Module):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__()      
        self.input_dim = dims['input_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.output_dim = dims['output_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.vocab_size = dims['vocab_size']
        self.linear_in = nn.Linear(self.input_dim, self.vocab_size)
        self.linear_out = nn.Linear(self.vocab_size, self.output_dim)

    def loss(self, preds, label_ids, ignore_index=0):
        return nn.CrossEntropyLoss(ignore_index=ignore_index)(preds.permute(0, 2, 1), label_ids)
    
    def embed_from_discrete_representation(self, x):
        return self.linear_out(x)
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def discretize(self, x) -> dict:
        pass

    @abstractmethod
    def embed_from_id(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass