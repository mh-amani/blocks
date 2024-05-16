from .abstract import AbstractDataset, AbstractPLDataModule
import datasets
import re
from datasets import load_dataset, DatasetDict
from torch.utils.data import random_split, Subset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional
import hydra
# from fairseq import tokenizer



class PCFGSetDataset(AbstractDataset):
    """
    A torch.utils.data.Dataset of PCFG Set dataset
    """
    # class variable to track whether train-val split has been performed - it should be done only once
    has_split = False  

    def __init__(self, dataset_parameters, **kwargs):
    
        super().__init__(dataset_parameters, **kwargs)
        self.params = kwargs
        self.seed = dataset_parameters['seed']
        self.dataset_parameters = dataset_parameters
        self.split = self.params['split']
        assert self.split in {"train", "val", "test"}, "Unexpected split reference"
        

        if not PCFGSetDataset.has_split:  # perform split if it has not been done yet
            PCFGSetDataset.overfit_batch = dataset_parameters['overfit_batch']
            
            # load the src and trg texts from the path for train and test splits
            dataset_dict_keys = ['train', 'test']
            self.loaded_dataset = DatasetDict({})
            for key in dataset_dict_keys:
                src_path = self.dataset_parameters[key]['src_path']
                tgt_path = self.dataset_parameters[key]['tgt_path']
                src = open(src_path, 'r').readlines()
                tgt = open(tgt_path, 'r').readlines()


                for i, line in enumerate(src):
                    # inject spaces between any letter that is immediately followed by a number
                    # this is to make sure that tokens like A15 are decomposed to A and 15
                    # line = re.sub(r'([a-zA-Z]+)([0-9]+)', r'\1 \2', line)

                    # A15 --> A 1 5
                    line = re.sub(r'([a-zA-Z]+)|([0-9]+)', lambda match: (match.group(1) + ' ') if match.group(1) else ' '.join(match.group(2)), line)
                    # line = re.sub(r'([a-zA-Z]+)([0-9]+)', r'\1 \2', line)
                    src[i] = line

                for i, line in enumerate(tgt):
                    line = re.sub(r'([a-zA-Z]+)|([0-9]+)', lambda match: (match.group(1) + ' ') if match.group(1) else ' '.join(match.group(2)), line)
                    # line = re.sub(r'([a-zA-Z]+)([0-9]+)', r'\1 \2', line)
                    tgt[i] = line
                
                self.loaded_dataset[key] = datasets.Dataset.from_dict({'source': src, 'target': tgt})
            
            self.train_ratio = dataset_parameters['train_ratio']
            
            # overfit batch, using small batch of same data for training and validation
            if PCFGSetDataset.overfit_batch:
                self.train_dataset, self.val_dataset = self.set_same_batch(self.loaded_dataset, PCFGSetDataset.overfit_batch)
                self.test_dataset = self.val_dataset
            else:
                self.train_dataset, self.val_dataset = self.split_train_val(self.loaded_dataset)
                self.test_dataset = self.loaded_dataset['test']

            PCFGSetDataset.datum = {}
            PCFGSetDataset.datum['train'] = self.train_dataset
            PCFGSetDataset.train_len = len(self.train_dataset)
            PCFGSetDataset.datum['val'] = self.val_dataset
            PCFGSetDataset.datum['test'] = self.test_dataset

            PCFGSetDataset.has_split = True  # mark that the split has been performed

            self.assign_data_type(PCFGSetDataset.train_len)

        self.data = PCFGSetDataset.datum[self.split]
        self.data_type = PCFGSetDataset.data_type[self.split]
        

    def split_train_val(self, loaded_dataset):
        assert (self.train_ratio > 0 and self.train_ratio < 1), "Unexpected train_test ratio" 
        train_size = int(self.train_ratio * len(loaded_dataset['train']))
        lengths = [train_size, len(loaded_dataset['train']) - train_size]
        train_dataset, val_dataset = random_split(loaded_dataset['train'], lengths, 
                                                  generator=torch.Generator().manual_seed(self.seed))
        return train_dataset, val_dataset 


    def set_same_batch(self, dataset, batchsize):
        return Subset(dataset['train'], range(batchsize)), Subset(dataset['train'], range(batchsize))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # data_type should be [X_available, Z_available], where 1 means available and 0 means unavailable
        return {"id": idx, "x": self.data[idx]['source'], 
                "z": self.data[idx]['target'], 'data_type': self.data_type[idx]}


class PCFGSetDatamodule(AbstractPLDataModule):
    """
    A pytorch-lightning DataModule for PCFG Set dataset
    """
    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(**kwargs)
        self.dataset_parameters = dataset_parameters
        
        self.data_train = hydra.utils.instantiate(self.params['datasets']['train'], self.dataset_parameters)
        self.data_val = hydra.utils.instantiate(self.params['datasets']['val'], self.dataset_parameters)
        # self.data_test = hydra.utils.instantiate(self.params['datasets']['test'], self.dataset_parameters)
        self.data_test = self.data_val

    