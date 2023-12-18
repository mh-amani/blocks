from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
from torch import nn
import numpy as np
# from vector_quantize_pytorch import VectorQuantize
import math


class VQVAEDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)      
        
        # self.logit_scale = torch.tensor((self.dictionary_dim * 2)**0.5, requires_grad=False)
        self.logit_scale = torch.tensor((2)**0.5, requires_grad=False)
        self.logit_init = nn.Parameter(torch.ones([]) * np.log(self.dictionary_dim), requires_grad=False)
        
        self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        torch.nn.init.uniform_(self.dictionary.weight, -1, 1)
        self.output_embedding = nn.Linear(self.output_dim, self.dictionary_dim, bias=True)
        torch.nn.init.normal_(self.output_embedding.weight, mean=0, std=1/ 3 / math.sqrt(self.output_dim))

        # self.output_embedding = lambda x: x
        self.encoder_embedding = nn.Linear(self.dictionary_dim, self.input_dim, bias=False)
        torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.encoder_embedding = lambda x: x
        self.decoder_embedding = nn.Linear(self.dictionary_dim, self.output_dim, bias=False)
        torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=1/math.sqrt(self.dictionary_dim))
        # self.decoder_embedding = lambda x: x

        self.dist_ord = kwargs.get('dist_ord', 2) 
        self.embedding_loss_torch = torch.nn.functional.mse_loss # torch.nn.CosineSimilarity(dim=-1)
        self.hard = kwargs['hard']
        self.kernel = nn.Softmax(dim=-1)
        self.beta = kwargs.get("beta",0.25) #0.25 is the beta used in the vq-vae paper
        
    ###################
    #Probably can remove these as we are using th matrix projection now
    # def fetch_embeddings_by_index(self,indices):
    #     if self.normalize_embeddings:
    #         return nn.functional.normalize(self.dictionary(indices),dim=-1)
    #     #~else
    #     return self.dictionary(indices)
        
    # def fetch_embedding_matrix(self):
    #     if self.normalize_embeddings:
    #         return nn.functional.normalize(self.dictionary.weight,dim=-1)
    #     #~else
    #     return self.dictionary.weight
    ###################
    def embedding_loss(self, quantized, x):
        return(self.embedding_loss_torch(quantized, x, reduction='none').mean(dim=-1))
    
    def discretize(self, x, **kwargs) -> dict: 
        dists = self.codebook_distances(x)
        logits = (1 - dists / self.logit_scale  - self.logit_init) / self.temperature   
        # logits = - dists/self.logit_scale * self.logit_init * self.temperature
        probs = self.kernel(logits)
        indices = torch.argmax(probs, dim=-1)

        if self.hard:
            # Apply STE for hard quantization
            quantized_dict_only = self.dictionary(indices)#self.fetch_embeddings_by_index(indices)
            quantized = quantized_dict_only + x - (x).detach()
        else:
            quantized = torch.matmul(probs,  self.dictionary.weight)

        if kwargs.get("supervision",False):
            true_quantized = self.dictionary(kwargs.get("true_ids",None))
            commitment_loss = self.embedding_loss(true_quantized.detach(), x)
            embedding_loss = self.embedding_loss(true_quantized, x.detach())
            
        else:
            commitment_loss = self.embedding_loss(quantized.detach(), x) 
            embedding_loss = self.embedding_loss(quantized_dict_only, x.detach())
            
        vq_loss = self.beta * commitment_loss + embedding_loss
        if torch.isnan(vq_loss).any():
            print("Loss is NaN. Adding breakpoint.")
            breakpoint()
        
        return indices, probs, logits, quantized, vq_loss

    def codebook_distances(self, x):
        
        #dictionary_expanded = self.fetch_embedding_matrix().unsqueeze(0).unsqueeze(1) # Shape: (batch, 1, vocab, dim)
        dictionary_expanded = self.dictionary.weight.unsqueeze(0).unsqueeze(1)
        x_expanded = x.unsqueeze(2)
        # if self.normalize_embeddings:
        #     x_expanded = nn.functional.normalize(x,dim=-1).unsqueeze(2)  # Shape: (batch, length, 1, dim)
        # else:   
        #     x_expanded = x.unsqueeze(2)  # Shape: (batch, length, 1, dim)

        # Compute the squared differences
        dist = torch.linalg.vector_norm(x_expanded - dictionary_expanded, ord=self.dist_ord, dim=-1)
        if dist.isnan().any():
            print("Nan in distances")
            breakpoint()
        return dist

    