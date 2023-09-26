#!/bin/bash

#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=05:59:00
#SBATCH --error=./slurm_err/sym_ae_%j.err

#SBATCH --output=./slurm_out/sym_ae_%j.out
# module load miniconda/3
# conda activate blocks

# wandb login

# export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
# # export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN ---------------------------------------------- #
# supervised:
# export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.02-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] +test=true || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.04-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.04, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] +test=true || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.06-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.06, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] +test=true || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.08-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.08, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] +test=true || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.01-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.01, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] +test=true || true


python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.2-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.2, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.3-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.3, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.4-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.4, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true


# test the supervised data:
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.02-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.04-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.04, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.06-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.06, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.08-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.08, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
# python3 run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.01-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.01, 0.9]" trainer.devices=[1] model.optimizer.lr=0.002 trainer.min_epochs=200 trainer.min_epochs=200 datamodule.dataset_parameters.batch_size=512 datamodule.dataset_parameters.train_ratio=0.99 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true


# weakly supervised:
# python3 run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.01-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.01, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.002 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=256 +model.substitute_config.model_params.max_x_length=20 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.01-gumbel/2023-09-24_21-21-14/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] || true
# python3 run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.02-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.002 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=128 +model.substitute_config.model_params.max_x_length=20 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.02-gumbel/2023-09-23_23-39-48/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] 
# python3 run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.04-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.04, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.002 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=256 +model.substitute_config.model_params.max_x_length=20 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.04-gumbel/2023-09-24_12-03-07/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] || true
# python3 run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.06-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.06, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.002 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=256 +model.substitute_config.model_params.max_x_length=20 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.06-gumbel/2023-09-24_20-33-12/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] || true
# python3 run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.08-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.08, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.002 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=256 +model.substitute_config.model_params.max_x_length=20 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.08-gumbel/2023-09-22_18-27-57/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"]



# testing


# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- sfst ---------------------------------------------- #
# supervised:
python3 run_train.py +experiment=sfst_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.3-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.3, 0.9]" trainer.devices=[1] trainer.min_epochs=100 model.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
python3 run_train.py +experiment=sfst_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.35-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.35, 0.9]" trainer.devices=[1] trainer.min_epochs=100 model.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] || true
python3 run_train.py +experiment=sfst_gpt2-gpt2_gumbel_supervised.yaml run_name="supervised-only-0.4-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.4, 0.9]" trainer.devices=[1] trainer.min_epochs=100 model.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=512 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"]

# weakly supervised:
# python3 run_train.py +experiment=sfst_pretrained_weaksup.yaml run_name="supervised-curriculum-0.1-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.1, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.001 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=128 +model.substitute_config.model_params.max_x_length=60 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.1-gumbel/2023-09-23_22-17-03/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] 
# python3 run_train.py +experiment=sfst_pretrained_weaksup.yaml run_name="supervised-curriculum-0.15-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.15, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.001 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=128 +model.substitute_config.model_params.max_x_length=60 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.15-gumbel/2023-09-23_23-34-47/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"]
# python3 run_train.py +experiment=sfst_pretrained_weaksup.yaml run_name="supervised-curriculum-0.25-gumbel" datamodule.dataset_parameters.supervision_ratio="[0.25, 0.9]" trainer.devices=[1] model.substitute_config.optimizer.lr=0.001 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.batch_size=128 +model.substitute_config.model_params.max_x_length=60 +model.substitute_config.model_params.max_z_length=60 model.checkpoint_path="/home/masani/blocks/logs/training/runs/supervised-only-0.25-gumbel/2023-09-24_12-09-41/checkpoints/last.ckpt" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"]  
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set -------------------------------------------- #

# -------------------------- Mixed Training -------------------------- #
# systematicitiy
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised +datamodule/experiment="systematicitiy" trainer.accelerator='gpu' trainer.devices=1 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["mixed-training"]

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
# python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised trainer.accelerator='gpu' datamodule.dataset_parameters.supervision_ratio="[0.01, 0.99]" logger.wandb.tags=["cogs","mixed-training"]

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ----------------------------------------------- #

# -------------------------- Mixed Training -------------------------- #
# data_type_sampling_probability=[0.8, 0.10, 0.10], supervision_ratio: [0.05, 0.45, 0.55]
# use BPE tokenizer
# python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised trainer.accelerator='gpu' trainer.devices=1 datamodule.dataset_parameters.supervision_ratio="[0.05, 0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["mixed-training"]

# deactivate
# module purge
