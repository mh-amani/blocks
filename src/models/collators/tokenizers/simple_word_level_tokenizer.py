from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader

import os


# def batch_iterator(dataset, key, batch_size=1000):
#     for i in range(0, len(dataset), batch_size):
#         yield dataset[i : i + batch_size][key]


# def SimpleWordLevelTokenizer(dataset, **kwargs):
#     key = kwargs['key']
#     tokenizer = Tokenizer(models.WordLevel(unk_token="[unk]"))
#     tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
#     iterator = batch_iterator(dataset, key=key, batch_size=kwargs['batch_size'])
    
#     trainer = trainers.WordLevelTrainer(special_tokens=list(kwargs['special_tokens']), vocab_size=kwargs['max_vocab_size'])
#     tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(dataset))
    
#     tokenizer.save('./tokenizer_'+ key + '.json')
    
#     return tokenizer

def batch_iterator(loader, key):
    for batch in loader:
        yield batch[key]

    


def SimpleWordLevelTokenizer(dataset, **kwargs):
    key = kwargs['key']
    tokenizer = Tokenizer(models.WordLevel(unk_token="[unk]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Create a DataLoader with the given batch size
    loader = DataLoader(dataset, batch_size=kwargs['batch_size'], shuffle=False)
    iterator = batch_iterator(loader, key=key)
    # Initialize trainer outside of the loop
    trainer = trainers.WordLevelTrainer(special_tokens=list(kwargs['special_tokens']), vocab_size=kwargs['max_vocab_size'])
    
    # Train tokenizer on the current batch
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(dataset))

    # Save the tokenizer
    tokenizer.save('./tokenizer_' + key + '.json')

    return tokenizer


def SimpleUnigramTokenizer(dataset, **kwargs):
    key = kwargs['key']
    tokenizer = Tokenizer(models.BPE(unk_token="[unk]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    iterator = batch_iterator(dataset, key=key, batch_size=kwargs['batch_size'])
    
    trainer = trainers.BpeTrainer(special_tokens=list(kwargs['special_tokens']), vocab_size=kwargs['max_vocab_size'])
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(dataset))

    tokenizer.save('./tokenizer_'+ key + '.json')

    return tokenizer


def PretrainedTokenizer(**kwargs):
    key = kwargs['key']
    if kwargs['tokenizer_path'] == -1:
        model_path = kwargs['checkpoint_path']
        path = os.path.dirname(os.path.dirname(model_path)) + '/tokenizer_'+ key + '.json'
    tokenizer = Tokenizer.from_file(path)
    tokenizer.save('./tokenizer_'+ key + '.json')
    return tokenizer


# def SimpleUnigramTokenizer(dataset, key, **kwargs):
#     tokenizer = Tokenizer(models.Unigram())
#     trainer = trainers.UnigramTrainer(
#         vocab_size=kwargs['vocab_size'],
#         special_tokens=list(kwargs['special_tokens'])
#     )
#     iterator = batch_iterator(dataset, key=key, batch_size=kwargs['batch_size'])

#     tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(dataset))

#     return tokenizer