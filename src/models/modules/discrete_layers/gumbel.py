from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax


class GumbelDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.tau = kwargs['tau']
        self.hard = kwargs['hard'] # if True, use argmax in forward pass, else use gumbel softmax. the backwardpass is the same in both cases

    def forward(self, x):
        x_in = self.linear_in(x)
        x_probs = gumbel_softmax(x_in, tau=self.tau, hard=self.hard, dim=-1)
        x_out = self.linear_out(x_probs)
        return x_out

    def discretize(self, x) -> dict:
        x_in = self.linear_in(x)
        x_probs = gumbel_softmax(x_in, tau=self.tau, hard=self.hard, dim=-1)
        return x_probs

    def embed_from_id(self, x):
        classes = torch.eye(self.vocab_size, device=x.device)[x]
        return self.linear_out(classes)
    
    def decode(self, x):
        return torch.argmax(x, dim=-1)