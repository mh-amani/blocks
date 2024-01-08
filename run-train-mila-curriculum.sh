#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=40G
#SBATCH --time=23:59:00
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
SEQMODEL='bart'
CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-softmax_continous/2023-12-31_12-08-58/checkpoints/last.ckpt"


# curriculum, 1 gpu:
python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.substitute_config.optimizer.lr=0.0001

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=32
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
CKPT='.../checkpoints/last.ckpt'


# curriculum, 1 gpu:
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True model.optimizer.lr=0.0001



# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=128
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
CKPT='.../checkpoints/last.ckpt'

# curriculum, 1 gpu:
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True model.optimizer.lr=0.0001


deactivate
module purge
