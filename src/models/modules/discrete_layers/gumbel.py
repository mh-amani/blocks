from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
from torch.nn.functional import gumbel_softmax


class GumbelDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.hard = kwargs['hard'] # if True, use argmax in forward pass, else use gumbel softmax. the backwardpass is the same in both cases
        self.temperature = kwargs['tau']

    def discretize(self, x) -> dict:
        x_probs = gumbel_softmax(x, tau=self.temperature, hard=self.hard, dim=-1)
        return x_probs
    
    def decode(self, x):
        # return gumbel_softmax(x, tau=self.temperature, hard=True, dim=-1).argmax(dim=-1)
        # this is wrong, the softmax should not applied to the scores, but to the logits.
        
        # return gumbel_softmax(torch.log(x), tau=self.temperature, hard=True, dim=-1).argmax(dim=-1)
        return torch.argmax(x, dim=-1)
        
        # return torch.argmax(x, dim=-1)
    