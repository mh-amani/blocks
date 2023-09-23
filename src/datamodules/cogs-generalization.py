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
import pandas as pd



class COGSDataset(AbstractDataset):
    """
    A torch.utils.data.Dataset of COGS Set dataset
    """
    # class variable to track whether train-val split has been performed - it should be done only once
    has_split = True  

    def __init__(self, dataset_parameters, **kwargs):
    
        super().__init__(dataset_parameters, **kwargs)
        self.params = kwargs
        self.seed = dataset_parameters['seed']
        self.dataset_parameters = dataset_parameters
        self.split = self.params['split']
        assert self.split in {"train", "val", "test", "gen"}, "Unexpected split reference"
        

        
        COGSDataset.overfit_batch = dataset_parameters['overfit_batch']
        
        # load the src and trg tsv files from the path for each split
        dataset_dict_keys = ['train', 'val', 'test', 'gen']
        self.loaded_dataset = DatasetDict({})
        for key in dataset_dict_keys:
            tsv_path = self.dataset_parameters[key]['tsv_path']

            # read the first two columns of the .tsv into a dataframe with columns ['source', 'target']
            # include the first row as a row in the dataframe, not as the column titles
            df = pd.read_csv(tsv_path, sep='\t', header=None, names=['source', 'target'], usecols=[0,1])
            # create a hf dataset from the dataframe
            dataset = datasets.Dataset.from_pandas(df) # dataset: Dataset({features: ['source', 'target'], num_rows: 24155})
            self.loaded_dataset[key] = dataset
                
        # overfit batch, using small batch of same data for training and validation
        if COGSDataset.overfit_batch:
            self.train_dataset, self.val_dataset = self.set_same_batch(self.loaded_dataset, COGSDataset.overfit_batch)
            self.test_dataset = self.val_dataset
        else:
            self.train_dataset, self.val_dataset = self.loaded_dataset['train'], self.loaded_dataset['val']
            # TODO: we might want to change this to the test set to test in-distribution performance
            self.test_dataset = self.loaded_dataset['gen']

        COGSDataset.datum = {}
        COGSDataset.datum['train'] = self.train_dataset
        COGSDataset.train_len = len(self.train_dataset)
        COGSDataset.datum['val'] = self.val_dataset
        COGSDataset.datum['test'] = self.test_dataset

        COGSDataset.has_split = True  # mark that the split has been performed

        self.assign_data_type(COGSDataset.train_len)

        self.data = COGSDataset.datum[self.split]
        self.data_type = COGSDataset.data_type[self.split]


    def set_same_batch(self, dataset, batchsize):
        return Subset(dataset['train'], range(batchsize)), Subset(dataset['train'], range(batchsize))


    def __len__(self):
        return len(self.data)

    def assign_data_type(self, train_length):
        """
        Add a supervision label to each data point in the train dataset
        """
        AbstractDataset.sup_len_xz = int(self.supervision_ratio[0] * train_length)
        AbstractDataset.sup_len_x = int( (1 - self.supervision_ratio[0]) * (1 - self.supervision_ratio[1]) * train_length)
        # data_type should be [X_available, Z_available], where 1 means available and 0 means unavailable
        AbstractDataset.data_type = {}
        AbstractDataset.data_type['train'] = np.ones((len(self.train_dataset), 2), dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_xz:AbstractDataset.sup_len_xz + AbstractDataset.sup_len_x] = np.array([1, 0], dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_xz + AbstractDataset.sup_len_x:] = np.array([0, 1], dtype=np.bool_)

        AbstractDataset.data_type['val'] = np.ones((len(self.val_dataset),2), dtype=np.bool_)
        AbstractDataset.data_type['test'] = np.ones((len(self.test_dataset),2), dtype=np.bool_)

    def __getitem__(self, idx):
        # data_type should be [X_available, Z_available], where 1 means available and 0 means unavailable
        return {"id": idx, "X": self.data[idx]['source'], 
                "Z": self.data[idx]['target'], 'data_type': self.data_type[idx]}


class COGSDatamodule(AbstractPLDataModule):
    """
    A pytorch-lightning DataModule for COGS Set dataset
    """
    def __init__(self, dataset_parameters, **kwargs):
        super().__init__(**kwargs)
        self.dataset_parameters = dataset_parameters
        
        self.data_train = hydra.utils.instantiate(self.params['datasets']['train'], self.dataset_parameters)
        self.data_val = hydra.utils.instantiate(self.params['datasets']['val'], self.dataset_parameters)
        self.data_test = hydra.utils.instantiate(self.params['datasets']['test'], self.dataset_parameters)

    