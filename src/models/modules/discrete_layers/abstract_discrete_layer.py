from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from torch.nn import LayerNorm,BatchNorm1d
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
        
        self.dictionary_weight_norm = kwargs.get('dictionary_weight_norm', False)
                
        if self.dictionary_weight_norm:
            self.dictionary = nn.utils.weight_norm(self.dictionary, dim=-1)
            breakpoint()

        self.output_embedding = nn.Linear(self.output_dim, self.dictionary_dim, bias=False)
        torch.nn.init.normal_(self.output_embedding.weight, mean=0, std=1/math.sqrt(self.output_dim))

        # self.output_embedding = lambda x: x
        self.encoder_embedding = nn.Linear(self.dictionary_dim, self.input_dim, bias=False)
        torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.encoder_embedding = lambda x: x
        self.decoder_embedding = nn.Linear(self.dictionary_dim, self.output_dim, bias=False)
        torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.decoder_embedding = lambda x: x
        
        
        self.bottleneck_normalization_args = kwargs.get('bottleneck_normalization_args', None)
        #weight norm must be done at initialization
        if  self.bottleneck_normalization_args.get("type",None) == 'weight norm':
            self.encoder_embedding = nn.utils.weight_norm(self.encoder_embedding, dim=-1)
            self.decoder_embedding = nn.utils.weight_norm(self.decoder_embedding, dim=-1)
            self.encoder_embedding_normalization = lambda x: x  #weight norm is done at initialization
            self.decoder_embedding_normalization = lambda x: x  #weight norm is done at initialization
        #otherwise, it's normalization is node in the Forward pass
        else:
            self.encoder_embedding_normalization = self.get_normalization_method(self.encoder_embedding, norm_args=self.bottleneck_normalization_args,output_dimension = self.dictionary_dim)
            self.decoder_embedding_normalization = self.get_normalization_method(self.decoder_embedding, norm_args=self.bottleneck_normalization_args,output_dimension = self.output_dim)
            
    def get_normalization_method(self,layer,norm_args,output_dimension):
        
        if norm_args is None:
            return layer
        
        method_type =  norm_args['type']
        method_args = norm_args['args']
        
        if method_type == 'layer norm':
            breakpoint()
            return LayerNorm(normalized_shape= output_dimension,**method_args)
        
        elif method_type == 'batch norm':
            return BatchNorm1d(num_features = output_dimension,**method_args)
        
        else:
            raise ValueError('normalization method not supported: {}'.format(method_type))        
    
    def decoder_to_discrete_embedding(self, x):
       out_x = self.output_embedding(x)
       return out_x
    
    def discrete_embedding_to_decoder(self, x):
        x = self.decoder_embedding(x)
        
        if self.bottleneck_normalization_args.get("type",None) == 'batch norm':
            return self.decoder_embedding_normalization(x.permute((0, 2, 1))).permute((0, 2, 1))
        
        return  self.decoder_embedding_normalization(x)
    
    def discrete_embedding_to_encoder(self, x):
        
        x = self.encoder_embedding(x)
        
        if self.bottleneck_normalization_args.get("type",None) == 'batch norm':
            return self.encoder_embedding_normalization(x.permute((0, 2, 1))).permute((0, 2, 1))
        
        return self.encoder_embedding_normalization(x)

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


    