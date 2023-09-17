import torch

def pad_logit_label(logits, labels, pad_token_id):
    if labels.shape[-1] > logits.shape[-2]:
        shape = list(logits.shape[0:-2]) + [labels.shape[-1] - logits.shape[-2]] + list(logits.shape[-1:])
        attached_vector = torch.ones(shape, device=logits.device, dtype=logits.dtype).fill_(-float('inf'))
        attached_vector[..., :, pad_token_id] = 1
        logits = torch.cat((logits, attached_vector), dim=-2)
    elif logits.shape[-2] >  labels.shape[-1]:
        shape = list(labels.shape[0:-1]) + [logits.shape[-2] - labels.shape[-1]] 
        attached_vector = torch.ones(shape, device=labels.device, dtype=labels.dtype).fill_(pad_token_id)
        labels = torch.cat((labels, attached_vector), dim=-1)
    
    return logits, labels

def pad_label_label(label1, label2, pad_token_id):
    if label1.shape[-1] > label2.shape[-1]:
        attached_vector = torch.ones_like(label1)[..., 0:label1.shape[-1] - label2.shape[-1]].fill_(pad_token_id)
        label2 = torch.cat((label2, attached_vector), dim=-1)
    elif label2.shape[-1] >  label1.shape[-1]:
        attached_vector = torch.ones_like(label2)[..., 0:label2.shape[-1] - label1.shape[-1]].fill_(pad_token_id)
        label1 = torch.cat((label1, attached_vector), dim=-1)
    return label1, label2
