from .abstract_discrete_layer import AbstractDiscreteLayer
import torch

class SoftmaxContinousLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)
        self.output_embedding = torch.nn.Linear(self.output_dim, self.vocab_size)

    def discretize(self, x,**kwargs) -> dict:
        score = torch.softmax(x/self.temperature, dim=-1)
        x_quantized = torch.matmul(score, self.dictionary.weight)
        id = torch.argmax(score, dim=-1)
        quantization_loss = 0
        return id, score, x_quantized, quantization_loss
    