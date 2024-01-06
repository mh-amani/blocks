from abc import ABC
from typing import Optional, Callable

import hydra
import torch

import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils.datamodule import randomSupervisionSampler, randomSupervisionSamplerDDP

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
        assert len(self.dataset_parameters["supervision_ratio"]) == 2, f"supervision_ratio should only contain 2 probabilities ([xz, p(z|not zx)]), got {len(self.dataset_parameters['supervision_ratio'])}"
        self.supervision_ratio = torch.tensor(self.dataset_parameters["supervision_ratio"]).float()

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
        AbstractDataset.sup_len_xz = int(self.supervision_ratio[0] * train_length)
        AbstractDataset.sup_len_unsup = int( (1 - self.supervision_ratio[0]) * self.supervision_ratio[1] * train_length)

        idx_xz = np.random.choice(train_length, AbstractDataset.sup_len_xz, replace=False)
        idx_x = np.random.choice(train_length, AbstractDataset.sup_len_unsup, replace=False)
        idx_z = np.random.choice(train_length, AbstractDataset.sup_len_unsup, replace=False)
        
        trainset_xz = torch.utils.data.Subset(self.train_dataset, idx_xz)
        trainset_x = torch.utils.data.Subset(self.train_dataset, idx_x)
        trainset_z = torch.utils.data.Subset(self.train_dataset, idx_z)
        total_length = len(trainset_xz) + len(trainset_x) + len(trainset_z)
        trainset = torch.utils.data.ConcatDataset([trainset_xz, trainset_x, trainset_z])

        # data_type should be [X_available, Z_available], where 1 means available and 0 means unavailable
        AbstractDataset.data_type = {}
        AbstractDataset.data_type['train'] = np.ones((total_length, 2), dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_xz:AbstractDataset.sup_len_xz + AbstractDataset.sup_len_unsup] = np.array([1, 0], dtype=np.bool_)
        AbstractDataset.data_type['train'][AbstractDataset.sup_len_xz + AbstractDataset.sup_len_unsup:] = np.array([0, 1], dtype=np.bool_)

        AbstractDataset.data_type['val'] = np.ones((len(self.val_dataset),2), dtype=np.bool_)
        AbstractDataset.data_type['test'] = np.ones((len(self.test_dataset),2), dtype=np.bool_)
        return trainset

    def __getitem__(self, idx):
        # supervision can be X, Z or XZ
        return {"id": idx, "x": self.data[idx]['x'], "z": self.data[idx]['z'], 'data_type': self.data_type[idx]}



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
        assert len(kwargs["data_type_sampling_probability"]) == 2, f"data_type_sampling_probability should only contain 2 probabilities ([xz, p(z|not zx)]), got {len(kwargs['data_type_sampling_probability'])}"
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
        
        # removing the data points with x or z length > max_X_length or max_Z_length after tokenization
        if self.dataset_parameters["remove_long_data_points"]:
            self.remove_long_data_points()
        
        if self.dataset_parameters.get('print_max_lengths', False):
            self.print_max_lengths()
            
    def remove_long_data_points(self):
        self.train_data = []
        counter = 0
        for i in range(len(self.data_train)):
            collated_batch = self.collate_fn([self.data_train[i]], cut_to_max_length=False)
            if collated_batch['x_ids'].shape[1] > self.dataset_parameters["max_x_length"] or collated_batch['z_ids'].shape[1] > self.dataset_parameters["max_z_length"]:
                continue
            data_i = self.data_train[i]
            data_i['id'] = counter
            counter += 1
            self.train_data.append(data_i)
        
        self.data_train.datum['train'] = self.train_data

        self.val_data = []
        counter = 0
        for i in range(len(self.data_val)):
            collated_batch = self.collate_fn([self.data_val[i]], cut_to_max_length=False)
            if collated_batch['x_ids'].shape[1] > self.dataset_parameters["max_x_length"] or collated_batch['z_ids'].shape[1] > self.dataset_parameters["max_z_length"]:
                continue
            data_i = self.data_val[i]
            data_i['id'] = counter
            counter += 1
            self.val_data.append(data_i)

        self.data_val.datum['val'] = self.val_data

        self.test_data = []
        counter = 0
        for i in range(len(self.data_test)):
            collated_batch = self.collate_fn([self.data_test[i]], cut_to_max_length=False)
            if collated_batch['x_ids'].shape[1] > self.dataset_parameters["max_x_length"] or collated_batch['z_ids'].shape[1] > self.dataset_parameters["max_z_length"]:
                continue
            data_i = self.data_test[i]
            data_i['id'] = counter
            counter += 1
            self.test_data.append(data_i)

        self.data_test.datum['test'] = self.test_data
    
    def print_max_lengths(self):
        max_x_length = 0
        larger_than_max_x_length = 0
        max_z_length = 0
        larger_than_max_z_length = 0
        for i in range(len(self.data_train)):
            collated_batch = self.collate_fn([self.data_train[i]], cut_to_max_length=False)
            if collated_batch['x_ids'].shape[1] > max_x_length:
                max_x_length = collated_batch['x_ids'].shape[1]
            if collated_batch['x_ids'].shape[1] > self.dataset_parameters["max_x_length"]:
                larger_than_max_x_length += 1
            if collated_batch['z_ids'].shape[1] > max_z_length:
                max_z_length = collated_batch['z_ids'].shape[1]
            if collated_batch['z_ids'].shape[1] > self.dataset_parameters["max_z_length"]:
                larger_than_max_z_length += 1
        print(f"max_x_length before cut-off: {max_x_length}, max_z_length before cut-off: {max_z_length}")
        # percentage of data points that are cut-off
        print(f"percentage of x data points that are cut-off: {larger_than_max_x_length/len(self.data_train)}")
        print(f"percentage of z data points that are cut-off: {larger_than_max_z_length/len(self.data_train)}")

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.params.get('use_ddp', False):
            self.train_sampler = randomSupervisionSamplerDDP(
                dataset=self.data_train, data_type_sampling_probability=self.data_type_sampling_probability, 
                batch_size=self.dataset_parameters["batch_size"])
        else:
            self.train_sampler = randomSupervisionSampler(
                self.data_train, self.data_type_sampling_probability,  
                batch_size=self.dataset_parameters["batch_size"], generator=g)

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

    