from typing import Type,Sequence
import functools
from pytorch_lightning import Callback, LightningModule, Trainer
from src.schedulers.abstract_scheduler import AbstractScheduler
import wandb
import torch

class SupervisionProbabilitySchedulerCallback(Callback):
    """
    PL Callback to anneal a hyperparameter during the training of the model (e.g. temperature of the gumbel softmax)
    """

    def __init__(self, scheduler_xz: Type[AbstractScheduler], scheduler_z: Type[AbstractScheduler]):
        """
        Args:
            hyperparameter_location: path to our hyperparameter in our model
            scheduler: type of scheduler we want to use
        """
        super().__init__()      
        self.sampler = None
        self.scheduler_xz = scheduler_xz
        self.scheduler_z = scheduler_z


    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
        ) -> None:

        if self.sampler==None:
            self.sampler = trainer.datamodule.train_sampler
            
        probs = self.sampler.data_type_sampling_probability
        
        prob_xz = self.scheduler_xz.update(probs[0], pl_module.current_epoch)
        prob_z = self.scheduler_z.update(probs[1], pl_module.current_epoch)

        new_hyperparameter_value = torch.tensor([prob_xz, prob_z])
        self.sampler.data_type_sampling_probability = new_hyperparameter_value
        
        pl_module.log_dict({'prob_xz': prob_xz, 'prob_z':prob_z,  'global_step': float(pl_module.global_step)})
        # wandb.log({self.hyperparameter_name: new_hyperparameter_value, 'global_step': pl_module.global_step})
        
# trainer.train_dataloader.sampler.p_sup