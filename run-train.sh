#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=05:59:00
#SBATCH --output=./slurm_out/sym_ae_%j.out
#SBATCH --error=./slurm_err/sym_ae_%j.err

module load miniconda/3
conda activate blocks

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
# export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN ---------------------------------------------- #
# supervised:
# python run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.02-gumbel" trainer.devices=[0] trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"]
# weakly supervised:
# python run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.02-gumbel" trainer.devices=[0] trainer.min_epochs=100 datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=9.0 logger.wandb.tags=["weakly-supervised"]

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set -------------------------------------------- #

# -------------------------- Mixed Training -------------------------- #
# systematicitiy
python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="systematicitiy" trainer.accelerator='gpu' trainer.devices=1 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.8 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["mixed-training"]

# substitutivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="substitutivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.8, 0.10, 0.10]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.05, 0.45, 0.55]" logger.wandb.tags=["mixed-training"]

# productivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="productivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.8, 0.10, 0.10]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.05, 0.45, 0.55]" logger.wandb.tags=["mixed-training"]

# pcfgset
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="pcfgset" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.8, 0.10, 0.10]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.05, 0.45, 0.55]" logger.wandb.tags=["mixed-training"]

# -------------------------- Weakly Supervised Training -------------------------- #
# data_type_sampling_probability=[0.98, 0.01, 0.01], supervision_ratio: [0.04, 0.95, 0.01]

# systematicitiy
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="systematicitiy" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.98, 0.01, 0.01]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.04, 0.95, 0.01]" logger.wandb.tags=["weakly"]

# substitutivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="substitutivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.98, 0.01, 0.01]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.04, 0.95, 0.01]" logger.wandb.tags=["weakly"]

# productivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="productivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.98, 0.01, 0.01]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.04, 0.95, 0.01]" logger.wandb.tags=["weakly"]

# pcfgset
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="pcfgset" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.98, 0.01, 0.01]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.04, 0.95, 0.01]" logger.wandb.tags=["weakly"]

# -------------------------- ZxZ only -------------------------- #
# data_type_sampling_probability=[0.01, 0.01, 0.98], supervision_ratio: [0.01, 0.01, 0.98]
# model.model_params.loss_coeff.xzx=-1.0, model.model_params.loss_coeff.supervised_seperated_x=-1.0, model.model_params.loss_coeff.supervised_seperated_z=-1.0
# systematicitiy
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="systematicitiy" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.01, 0.01, 0.98]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.01, 0.01, 0.98]" model.model_params.loss_coeff.xzx=-1.0 model.model_params.loss_coeff.supervised_seperated_x=-1.0 model.model_params.loss_coeff.supervised_seperated_z=-1.0 logger.wandb.tags=["ZxZ"]

# substitutivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="substitutivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.01, 0.01, 0.98]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.01, 0.01, 0.98]" model.model_params.loss_coeff.xzx=-1.0 model.model_params.loss_coeff.supervised_seperated_x=-1.0 model.model_params.loss_coeff.supervised_seperated_z=-1.0 logger.wandb.tags=["ZxZ"]

# productivity
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="productivity" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.01, 0.01, 0.98]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.01, 0.01, 0.98]" model.model_params.loss_coeff.xzx=-1.0 model.model_params.loss_coeff.supervised_seperated_x=-1.0 model.model_params.loss_coeff.supervised_seperated_z=-1.0 logger.wandb.tags=["ZxZ"]

# pcfgset
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised ++datamodule/experiment="pcfgset" trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.01, 0.01, 0.98]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.01, 0.01, 0.98]" model.model_params.loss_coeff.xzx=-1.0 model.model_params.loss_coeff.supervised_seperated_x=-1.0 model.model_params.loss_coeff.supervised_seperated_z=-1.0 logger.wandb.tags=["ZxZ"]


# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ---------------------------------------------- #

# -------------------------- Mixed Training -------------------------- #
# data_type_sampling_probability=[0.8, 0.10, 0.10], supervision_ratio: [0.05, 0.45, 0.55]
# use BPE tokenizer
python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised trainer.accelerator='gpu' trainer.devices=1 datamodule.data_type_sampling_probability="[0.8, 0.2]" datamodule.dataset_parameters.supervision_ratio="[0.01, 0.99]" logger.wandb.tags=["cogs","mixed-training"]

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ----------------------------------------------- #

# -------------------------- Mixed Training -------------------------- #
# data_type_sampling_probability=[0.8, 0.10, 0.10], supervision_ratio: [0.05, 0.45, 0.55]
# use BPE tokenizer
python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised trainer.accelerator='gpu' trainer.devices=1 +experiment.datamodule.data_type_sampling_probability="[0.8, 0.10, 0.10]" +experiment.datamodule.dataset_parameters.supervision_ratio="[0.05, 0.45, 0.55]" logger.wandb.tags=["cfq","mixed-training"]

deactivate
module purge
