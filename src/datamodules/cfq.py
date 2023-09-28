from .abstract import AbstractDataset, AbstractPLDataModule
from datasets import load_dataset
from torch.utils.data import random_split, Subset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Optional
import hydra


class CFQDataset(AbstractDataset):
    """
    A torch.utils.data.Dataset of CFQ dataset
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
        

        if not CFQDataset.has_split:  # perform split if it has not been done yet
            self.test_split = dataset_parameters['test_split']
            CFQDataset.overfit_batch = dataset_parameters['overfit_batch']
            assert self.test_split in ['mcd1', 'mcd2', 'mcd3', 'question_complexity_split', 'question_pattern_split'
                                       , 'query_complexity_split', 'query_pattern_split', 'random_split']
            self.loaded_dataset = load_dataset("cfq", self.test_split) # DatasetDict({train: Dataset({features: ['question', 'query'],   num_rows: 95743}), test: Dataset({features: ['question', 'query'], num_rows: 11968})})
            
            self.train_ratio = dataset_parameters['train_ratio']

            # overfit batch, using small batch of same data for training and validation
            if CFQDataset.overfit_batch:
                self.train_dataset, self.val_dataset = self.set_same_batch(self.loaded_dataset, CFQDataset.overfit_batch)
                self.test_dataset = self.val_dataset
            else:
                self.train_dataset, self.val_dataset = self.split_train_val(self.loaded_dataset)
                self.test_dataset = self.loaded_dataset['test']

            CFQDataset.datum = {}
            CFQDataset.datum['train'] = self.train_dataset
            CFQDataset.train_len = len(self.train_dataset)
            CFQDataset.datum['val'] = self.val_dataset
            CFQDataset.datum['test'] = self.test_dataset

            CFQDataset.has_split = True  # mark that the split has been performed

            self.assign_data_type(CFQDataset.train_len)

        self.data = CFQDataset.datum[self.split]
        self.data_type = CFQDataset.data_type[self.split]
        

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
        return {"id": idx, "x": self.data[idx]['question'], 
                "z": self.data[idx]['query'], 'data_type': self.data_type[idx]}


class CFQDatamodule(AbstractPLDataModule):
    """
    A pytorch-lightning DataModule for CFQ dataset
    """
    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(**kwargs)
        self.dataset_parameters = dataset_parameters
        
        self.data_train = hydra.utils.instantiate(self.params['datasets']['train'], self.dataset_parameters)
        self.data_val = hydra.utils.instantiate(self.params['datasets']['val'], self.dataset_parameters)
        self.data_test = hydra.utils.instantiate(self.params['datasets']['test'], self.dataset_parameters)

    