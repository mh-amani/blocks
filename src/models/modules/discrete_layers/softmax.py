from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
import math

class SoftmaxContinousLayer(AbstractDiscreteLayer):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__(dims, **kwargs)

        # supervised setting that work
        self.dictionary_std = 1/math.sqrt(self.dictionary_dim)
        self.input_std = 1/math.sqrt(self.dictionary_dim)
        self.out_std = 1/math.sqrt(self.output_dim)

        self.dictionary = torch.nn.Embedding(self.vocab_size, self.dictionary_dim)
        torch.nn.init.normal_(self.dictionary.weight, mean=0, std=self.dictionary_std)

        self.output_embedding = torch.nn.Linear(self.output_dim, self.vocab_size, bias=False)
        torch.nn.init.normal_(self.output_embedding.weight, mean=0, std=self.out_std)
        # self.output_embedding = lambda x: x
        
        self.encoder_embedding = torch.nn.Linear(self.dictionary_dim, self.input_dim, bias=False)
        torch.nn.init.normal_(self.encoder_embedding.weight, mean=0, std=self.input_std)
        # self.encoder_embedding = lambda x: x
        self.decoder_embedding = torch.nn.Linear(self.dictionary_dim, self.output_dim, bias=False)
        torch.nn.init.normal_(self.decoder_embedding.weight, mean=0, std=self.input_std)
        # self.output_embedding = lambda x: x

        self.logit_std = math.sqrt(self.output_dim * self.out_std**2)
        self.logit_init = math.log(self.dictionary_dim)

    def discretize(self, x,**kwargs) -> dict:
        logits= x/self.logit_std / self.temperature
        score = torch.softmax(logits, dim=-1)
        id = torch.argmax(score, dim=-1)
        # x_quantized = torch.matmul(score, self.dictionary.weight)
        x_quantized = torch.matmul(score, self.dictionary.weight) - torch.matmul(score, self.dictionary.weight).detach() + self.dictionary(id)
        quantization_loss = torch.tensor(0.0).to(x.device)
        return id, score, logits, x_quantized, quantization_loss
    