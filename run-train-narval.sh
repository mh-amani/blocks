#!/bin/bash

#SBATCH --account=def-gdumas85
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --time=23:59:00
#SBATCH --output=./slurm_out/sym_ae_%j.out
#SBATCH --error=./slurm_err/sym_ae_%j.err

module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.4
module load arrow/13.0.0
module load python/3.10
module load httpproxy
source /home/aminm/symae/bin/activate

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

# export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
# export WANDB_MODE=offline

# model.collator.tokenizer.vocab_size, model.lr_scheduler.patience/cooldown, model.optimizer.lr
# other accounts: rrg-bengioy-ad, def-gdumas85

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
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp datamodule.dataset_parameters.num_workers=48 model.optimizer.lr=0.001 || true

# only zxz
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz ddp" trainer=ddp callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=64
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'

# supervised
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp datamodule.dataset_parameters.num_workers=48 model.optimizer.lr=0.001 || true

# only zxz
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz ddp" trainer=ddp callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=64
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'

# supervised
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true


deactivate
module purge
