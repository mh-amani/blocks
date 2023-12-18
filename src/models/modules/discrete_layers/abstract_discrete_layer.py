from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from torch.nn import LayerNorm
import math
class AbstractDiscreteLayer(nn.Module):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__()      
        self.input_dim = dims['input_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.output_dim = dims['output_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.vocab_size = dims['vocab_size']
        self.dictionary_dim = kwargs['dictionary_dim']

        self.temperature = kwargs.get('temperature', 1.0)
        self.label_smoothing_scale = kwargs.get('label_smoothing_scale', 0.001)

        self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        torch.nn.init.normal_(self.dictionary.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))

        self.output_embedding = nn.Linear(self.output_dim, self.dictionary_dim, bias=False)
        torch.nn.init.normal_(self.output_embedding.weight, mean=0, std=1/math.sqrt(self.output_dim))

        # self.output_embedding = lambda x: x
        self.encoder_embedding = nn.Linear(self.dictionary_dim, self.input_dim, bias=False)
        torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.encoder_embedding = lambda x: x
        self.decoder_embedding = nn.Linear(self.dictionary_dim, self.output_dim, bias=False)
        torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.decoder_embedding = lambda x: x
    
    def decoder_to_discrete_embedding(self, x):
       out_x = self.output_embedding(x)
       return out_x
    
    def discrete_embedding_to_decoder(self, x):
        return self.decoder_embedding(x)
    
    def discrete_embedding_to_encoder(self, x):
        return self.encoder_embedding(x)
    
    def project_embedding_matrix(self):
        self.dictionary.weight = torch.nn.Parameter(self.dict_project(self.dictionary.weight))
    
    def forward(self, x,**kwargs):
        continous_vector = self.decoder_to_discrete_embedding(x)
          
        # scores are between 0 and 1, and sum to 1 over the vocab dimension.
        id, score, logit, quantized_vector, quantization_loss  = self.discretize(continous_vector,**kwargs)
        return id, score, logit, quantized_vector, quantization_loss
    
    def embed_enc_from_id(self, x):
        embeds = self.dictionary(x)
        return self.discrete_embedding_to_encoder(embeds)
    
    def embed_dec_from_id(self, x):
        embeds = self.dictionary(x)
        return self.discrete_embedding_to_decoder(embeds)

    @abstractmethod
    def discretize(self, x,**kwargs) -> dict:
        pass


    