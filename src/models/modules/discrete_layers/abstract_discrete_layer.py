from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class AbstractDiscreteLayer(nn.Module):
    def __init__(self, dims, **kwargs) -> None:
        super().__init__()      
        self.input_dim = dims['input_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.output_dim = dims['output_dim'] # fed by the model, after x->z and z->x models are instantiated
        self.vocab_size = dims['vocab_size']
        self.temperature = kwargs.get('temperature', 1)
        self.label_smoothing_scale = kwargs.get('label_smoothing_scale', 0.001)
        self.encoder_embedding = nn.Linear(self.vocab_size, self.input_dim)
        self.decoder_embedding = nn.Linear(self.vocab_size, self.output_dim)
        self.output_embedding = nn.Linear(self.output_dim, self.vocab_size)

    def loss(self, preds, label_ids, ignore_index=0):
        # return nn.CrossEntropyLoss(ignore_index=ignore_index)(preds.permute(0, 2, 1), label_ids)
        smoothed_preds = (1 - self.label_smoothing_scale) * preds + self.label_smoothing_scale / self.vocab_size
        return nn.NLLLoss(ignore_index=ignore_index)(torch.log(smoothed_preds).permute(0, 2, 1), label_ids)
    
    def decoder_to_scores(self, x):
        return self.output_embedding(x)
    
    def scores_to_decoder(self, x):
        return self.decoder_embedding(x)
    
    def scores_to_encoder(self, x):
        return self.encoder_embedding(x)
    
    def forward(self, x):
        x_in = self.decoder_to_scores(x)
        # scores are between 0 and 1, and sum to 1 over the vocab dimension.
        x_scores = self.discretize(x_in)
        # x_ids are the argmax or sample of the scores, i.e. the predicted classes.
        x_ids = self.decode(x_scores).detach()
        return {'ids':x_ids, 'scores':x_scores}
    
    def embed_enc_from_id(self, x):
        classes = torch.eye(self.vocab_size, device=x.device)[x]
        return self.scores_to_encoder(classes)
    
    def embed_dec_from_id(self, x):
        classes = torch.eye(self.vocab_size, device=x.device)[x]
        return self.scores_to_decoder(classes)

    @abstractmethod
    def discretize(self, x) -> dict:
        pass

    @abstractmethod
    def decode(self, x):
        pass

    