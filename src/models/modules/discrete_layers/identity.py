from .abstract_discrete_layer import AbstractDiscreteLayer
import torch


class GumbelDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)

    def forward(self, x):
        x_in = self.linear_in(x)
        x_probs = x_in
        x_ids = torch.argmax(x_probs, dim=-1)
        x_out = self.linear_out(x_probs)
        return {'x_ids':x_ids, 'x_scores':x_probs, 'x_out':x_out}

    def discretize(self, x, hard=False, **kwargs) -> dict:
        x_in = self.linear_in(x)
        x_probs = x_in
        return x_probs

    def embed_from_id(self, x):
        classes = torch.eye(self.vocab_size, device=x.device)[x]
        return self.linear_out(classes)
    
    def decode(self, x):
        return torch.argmax(x, dim=-1)