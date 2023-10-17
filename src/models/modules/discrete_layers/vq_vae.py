from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
from torch import nn
# from vector_quantize_pytorch import VectorQuantize
from entmax import sparsemax

class VQVAEDiscreteLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)      
        
        self.dictionary = nn.Embedding(self.vocab_size, self.dictionary_dim)
        self.dist_ord = kwargs.get('dist_ord', 2) 
        self.hard = kwargs['hard']
        self.kernel = nn.Softmax(dim=-1)

    def discretize(self, x) -> dict:
        probs = self.kernel( - self.codebook_distances(x) / self.temperature)
        indices = torch.argmax(probs, dim=-1)
        if self.hard:
            # Apply STE for hard quantization
            quantized = self.dictionary(indices)
            quantized = quantized + x - (x).detach()
        else:
            quantized = torch.matmul(probs, self.dictionary.weight)

        commit_loss = 0
        
        return indices, probs, quantized, commit_loss

    def codebook_distances(self, x):
        x_expanded = x.unsqueeze(2)  # Shape: (batch, length, 1, dim)
        dictionary_expanded = self.dictionary.weight.unsqueeze(0).unsqueeze(1)  # Shape: (batch, 1, vocab, dim)
        # Compute the squared differences
        dist = torch.linalg.vector_norm(x_expanded - dictionary_expanded, ord=self.dist_ord, dim=-1)
        return dist