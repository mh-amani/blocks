from pytorch_lightning import Callback, LightningModule, Trainer
from typing import Sequence
import wandb

class ProbabilityLogger(Callback):  # pragma: no cover
    """
    PL Callback to generate triplets of source, target, and prediction along with the bottleneck
    sequence (if any, i.e. a continuous seq2seq model does not have one).
    """

    def __init__(self, logging_batch_interval=100):
        """
        Args:
            logging_batch_interval: How frequently to inspect/potentially plot something
        """
        super().__init__()
        self.logging_batch_interval = logging_batch_interval
        self.wandb_table = wandb.Table(columns=["Source", "Z_Probs","Step","Epoch"])

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if (batch_idx + 1) % self.logging_batch_interval != 0:
            return
        
        # z_probs = outputs.pop('z_probs')
        # sources = outputs.pop('source')
        
        z_probs = 'Hiii change me'
        sources = 'Hiii change me'
                

        for source, z_prob in zip(
                sources.detach().cpu().numpy(), z_probs.detach().cpu().numpy()
            ):
                self.wandb_table.add_data(
                    " ".join(str(source)), str(z_prob),str(trainer.global_step),str(trainer.current_epoch)
                )
        return
        
        
    def on_train_end(self,trainer,pl_module):
        wandb.log({"Z_probs": self.wandb_table})