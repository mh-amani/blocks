from abc import ABC
from typing import Optional, Callable

import hydra
import torch

import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.datamodule import randomSupervisionSampler

from src.utils.general import get_pylogger

log = get_pylogger(__name__)


class AbstractDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, dataset_parameters, **kwargs):
        super().__init__()
        self.dataset_parameters = dataset_parameters
        self.params = kwargs
        self.random_state = np.random.RandomState(self.dataset_parameters["seed"])

        self.supervision_ratio = torch.tensor(self.dataset_parameters["supervision_ratio"]).float()
        self.supervision_ratio = self.supervision_ratio / self.supervision_ratio.sum()

    def __len__(self):
        raise NotImplementedError()

    def _load_data(self):
        raise NotImplementedError()

    def get_random_sample(self):
        idx = self.random_state.randint(0, len(self.data))
        return self.data[idx]

    def get_random_subset(self, k):
        idxs = self.random_state.choice(len(self.data), k, replace=False)
        return [self.data[idx] for idx in idxs]

    def get_bootstrapped_data(self, seed):
        data = self.data
        num_datapoints = len(data)

        random_state = np.random.RandomState(seed)
        bootstrap_ids = random_state.choice(len(self.data), num_datapoints, replace=True)

        bootstrap_data = [data[i] for i in bootstrap_ids]
        return bootstrap_data
    
    def assign_data_type(self, train_length):
        """
        Add a supervision label to each data point in the train dataset
        """
        AbstractDataset.sup_len_x = int(self.supervision_ratio[0] * train_length)
        AbstractDataset.sup_len_z = int(self.supervision_ratio[1] * train_length)
        # data_type should be [X_available, Z_available], where 1 means available and 0 means unavailable
        AbstractDataset.data_type = {}
        AbstractDataset.data_type['train'] = np.ones((len(self.train_dataset), 2), dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_x:AbstractDataset.sup_len_x + AbstractDataset.sup_len_z] = np.array([1, 0], dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_x + AbstractDataset.sup_len_z:] = np.array([0, 1], dtype=np.bool_)

        AbstractDataset.data_type['val'] = np.ones((len(self.val_dataset),2), dtype=np.bool_)
        AbstractDataset.data_type['test'] = np.ones((len(self.test_dataset),2), dtype=np.bool_)

    def __getitem__(self, idx):
        # supervision can be X, Z or XZ
        return {"id": idx, "X": self.data[idx]['X'], "Z": self.data[idx]['Z'], 'data_type': self.data_type[idx]}



# class AbstractOutputDataset(AbstractDataset, ABC):
#     @staticmethod
#     def get_predictions(item, key="prediction", top_pred_only=True):
#         preds = item[key]

#         if top_pred_only and not isinstance(preds, str):
#             return preds[0]

#         return preds

#     @staticmethod
#     def get_targets(item, key="target", wrap_in_list=False):
#         tgts = item[key]

#         if wrap_in_list and not isinstance(tgts, list):
#             return [tgts]

#         return tgts


class AbstractPLDataModule(LightningDataModule, ABC):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(self, seed: int,collate_fn: Callable = None, num_workers: int = 0, **kwargs):
        """

        Parameters
        ----------
        seed : Random seed
        collate_fn : The collate function which is model specific
        num_workers : Setting num_workers as a positive integer will turn on multiprocess data loading with the specified number of loader worker processes

        kwargs: dataset specific parameters

        Returns
        -------
        An instance of the Grid dataset that extends pytorch_lightning.DataModule
        """
        super().__init__()
        self.datasets = kwargs["datasets"]
        self.params = kwargs
        
        self.collate_fn = collate_fn
        # self.tokenizer_x = hydra.utils.call(self.params["tokenizer"], _recursive_ = False)

        # Concerning the loaders
        self.num_workers = num_workers
        self.seed = seed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.data_type_sampling_probability = torch.tensor(kwargs["data_type_sampling_probability"]).float()

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        
        assert stage in set(["fit", "validate", "test", None])

        if self.data_train is None:
            self.data_train = hydra.utils.instantiate(
                self.dataset_parameters["train"]["dataset"])
            log.info("The train dataset has been loaded and has %d samples" % len(self.data_train))
            
        if self.data_val is None:
            self.data_val = hydra.utils.instantiate(
                self.dataset_parameters["val"]["dataset"])
            log.info("The validation dataset has been loaded and has %d samples" % len(self.data_val))

        if (stage == "test" or stage is None) and self.data_test is None:
            self.data_test = hydra.utils.instantiate(
                self.dataset_parameters["test"]["dataset"])
            log.info("The test dataset has been loaded and has %d samples" % len(self.data_test))

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_sampler = randomSupervisionSampler(
            self.data_train, self.data_type_sampling_probability, generator=g, 
            batch_size=self.dataset_parameters["batch_size"])
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.dataset_parameters["batch_size"],
            num_workers=self.dataset_parameters["num_workers"],
            collate_fn=self.collate_fn,
            sampler=self.train_sampler,
            drop_last=False,
        )
    
    def val_dataloader(self):
        dataloader = self._get_dataloader(
            data=self.data_val,
            dataloader_parameters=self.dataset_parameters,
            drop_last=False,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = self._get_dataloader(
            data=self.data_test,
            dataloader_parameters=self.dataset_parameters,
            drop_last=False,
            shuffle=False,
        )
        return dataloader
        

    def _get_dataloader(self, data, dataloader_parameters, drop_last, shuffle):
        return DataLoader(
            dataset=data,
            batch_size=dataloader_parameters["batch_size"],
            num_workers=dataloader_parameters["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=drop_last,
            shuffle=shuffle,
        )
    


    # def on_save_checkpoint(self, checkpoint):
    # def state_dict(self):
    #     # track whatever you want here
    #     state = {"current_train_batch_index": self.current_train_batch_index}
    #     return state

    # def load_state_dict(self, state_dict):
    #     # restore the state based on what you tracked in (def state_dict)
    #     self.current_train_batch_index = state_dict["current_train_batch_index"]

    