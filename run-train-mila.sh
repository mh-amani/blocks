#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=40G
#SBATCH --time=35:59:00
#SBATCH --output=./slurm_out/sym_ae_%j.out
#SBATCH --error=./slurm_err/sym_ae_%j.err

module load miniconda/3
conda activate blocks

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
# export WANDB_MODE=offline

# model.collator.tokenizer.vocab_size, model.lr_scheduler.patience/cooldown, model.optimizer.lr
# # SBATCH --gres=gpu:a100l:2 # SBATCH --constraint="dgx"
# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN --------------------------------------------------------------- #

DEVICE=0
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- sfst --------------------------------------------------------------- #
# # supervised:
DEVICE=0
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set ------------------------------------------------------------- #

DEVICE=2
BSIZE=64
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'

# supervised
# 1 gpu
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.01 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true

# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 || true

# only zxz
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE model.optimizer.lr=0.00001 +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=32
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised
# 1 gpu
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true

# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp  model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 model.model_params.usez=True model.model_params.loss_coeff.zxz=0.1 || true
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.90, 0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001,0.0005 model.model_params.usex=True model.model_params.loss_coeff.xzx=0.1 model.model_params.usez=True model.model_params.loss_coeff.zxz=0.1 || true


# only zxz
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' model.optimizer.lr=0.0001 || true



# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised
# 1 gpu
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.01 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true

# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.0001 || true
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.90] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.0005 model.model_params.usex=True model.model_params.loss_coeff.xzx=0.1 || true

# only zxz
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' model.optimizer.lr=0.0001 || true


deactivate
module purge
