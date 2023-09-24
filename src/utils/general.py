import logging
import os
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only



def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)

import logging
import sys
import numpy as np

from pytorch_lightning.utilities import rank_zero_only

def data_type_str(data_type):
    out = ''
    if data_type[0]:
        out += 'X'
    if data_type[1]:
        out += 'Z'
    return out
    

def get_pylogger(name=__name__, stdout=False) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    if stdout:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)

@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "hydra",
        "logger",
        "debug",
        "seed",
        "run_name",
        "ignore_warnings",
        "test_after_training",
    ),
    keys_to_ignore: Sequence[str] = (
        # "trainer",
        # "model",
        # "datamodule",
        # "callbacks",
        "hydra",
        "logger",
        # "debug",
        # "seed",
        # "run_name",
        # "ignore_warnings",
        # "test_after_training",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from configs will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        if field in keys_to_ignore:
            continue
        else:
            branch = tree.add(field, style=style, guide_style=style)

            config_section = config.get(field)
            branch_content = str(config_section)
            if isinstance(config_section, DictConfig):
                branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

            branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)
        

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[Logger],
) -> None:
    
    """This method controls which parameters from Hydra configs are saved by Lightning loggers.
    Additionally saves:
        - number of trainable model parameters
    """

    # logger: List[pl.loggers.LightningLoggerBase],

    hparams = {}

    # choose which parts of hydra configs will be saved to loggers
    hparams["trainer"] = OmegaConf.to_container(config, resolve=True) 
    hparams["model"] = OmegaConf.to_container(config["model"], resolve=True)
    hparams["datamodule"] = OmegaConf.to_container(config["datamodule"], resolve=True)

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # save the relative path to the working directory
    from hydra.utils import get_original_cwd
    hparams["rel_path_to_work_dir"] = os.path.relpath(os.getcwd(), get_original_cwd())

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = OmegaConf.to_container(config["callbacks"], resolve=True)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""

    # List[pl.loggers.LightningLoggerBase]
    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger) or lg == "wandb_standalone":
            import wandb

            wandb.finish()





