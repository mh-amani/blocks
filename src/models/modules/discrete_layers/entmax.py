from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
import torch.nn as nn
from entmax import sparsemax, entmax15, entmax_bisect


class EntmaxDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, pad_token_id, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        # self.input_dim = self.hparams.input_dim
        self.input_dim = dims['input_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.output_dim = dims['output_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.vocab_size = dims['vocab_size']
        self.pad_token_id = pad_token_id
        self.alpha = kwargs['alpha']
        self.linear_in = nn.Linear(self.input_dim, self.vocab_size)
        self.linear_out = nn.Linear(self.vocab_size, self.output_dim)


    def forward(self, x):
        x_in = self.linear_in(x)
        x_probs = entmax_bisect(x_in, alpha=self.alpha, dim=-1)
        x_out = self.linear_out(x_probs)
        return x_out
    

    def loss(self, preds, label_ids):
        return nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(preds.permute(0, 2, 1), label_ids)


    def discretize(self, x) -> dict:
        x_in = self.linear_in(x)
        x_probs = entmax_bisect(x_in, alpha=self.alpha, dim=-1)
        return x_probs
    
    
    def embed_from_id(self, x):
        classes = torch.eye(self.vocab_size, device=x.device)[x]
        return self.linear_out(classes)
    
    def embed_from_discrete_representation(self, x):
        return self.linear_out(x)
    
    def decode(self, x):
        return torch.argmax(x, dim=-1)