import torch
from typing import Iterable
import numpy as np
import hydra
from omegaconf import DictConfig


class SimpleTextCollator:
    def __init__(self,
                data_train,
                max_X_length: int = None,
                max_Z_length: int = None,
                padding: bool = True,
                pad_token_id: int = None,
                **kwargs
            ):
        
       
        
        self.max_X_length = max_X_length
        self.max_Z_length = max_Z_length
        self.padding = padding
        self.pad_token_id = pad_token_id
        self.tokenize_prior_training = kwargs['tokenizer']['tokenize_prior_training']
        
        
        self.tokenizer_x = hydra.utils.instantiate(kwargs['tokenizer'], dataset=data_train, key='X', _recursive_=False)
        self.tokenizer_z = hydra.utils.instantiate(kwargs['tokenizer'], dataset=data_train, key='Z', _recursive_=False)

        if self.tokenize_prior_training:
            self.x_ids = [self.tokenizer_x.encode(sample['X']).ids for sample in data_train]
            self.z_ids = [self.tokenizer_z.encode(sample['Z']).ids for sample in data_train]

    def collate_fn(self, batch: Iterable[dict]):
        """
        A model specific collate function that will be passed to the datamodule i.e. the dataloaders.

        Parameters
        ----------
        batch : Is an iterable of elements returned by the getter of a training dataset (an
        instance of torch.utils.data.Dataset)

        Returns
        -------
        The input samples processed to the suitable format that will be passed to the forward method
        of the model by the dataloader.
        """

        collated_batch = {}

        key = "id"
        collated_batch["id"] = np.array([sample[key] for sample in batch], dtype=np.int_)

        if self.tokenize_prior_training:
            x_ids = [self.x_ids[i] for i in collated_batch["id"]]
            z_ids = [self.z_ids[i] for i in collated_batch["id"]]
        else:
            x_ids = [self.tokenizer_x.encode(sample[key]).ids for sample in batch]
            z_ids = [self.tokenizer_z.encode(sample[key]).ids for sample in batch]
        
        if self.max_X_length is not None:
            x_ids = [i[: self.max_X_length] for i in x_ids]
        
        if self.max_Z_length is not None:
            z_ids = [i[: self.max_Z_length] for i in z_ids]
            
        if self.padding:
            x_ids = self.pad(x_ids)
            z_ids = self.pad(z_ids)
            
        collated_batch["x_ids"] = torch.tensor(x_ids, dtype=torch.long)
        collated_batch["z_ids"] = torch.tensor(z_ids, dtype=torch.long)
        # super slow:
        collated_batch["data_type"] = np.prod([sample["data_type"] for sample in batch], dtype=np.int8, axis=0)

        return collated_batch
    

    def pad(self, data: Iterable[np.array]):
        max_len = max([len(arr) for arr in data])
        padded = np.array(
            [np.pad(arr, (0, max_len - len(arr)), "constant", constant_values=self.pad_token_id) for arr in data]
        )

        return padded
    


    
# def tokenize_dataset(self, dataset):
#         for i in range(len(dataset)):
#             dataset.tokenized_data['X'][i] = self.tokenizer_x.encode(dataset[i]['X']).ids
#             dataset.tokenized_data['Z'][i] = self.tokenizer_z.encode(dataset[i]['Z']).ids