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
# curriculum, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-softmax_continous/2023-12-31_12-08-58/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.16, 0.99]-bart-softmax_continous/2023-12-31_12-05-56/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.08, 0.99]-bart-softmax_continous/2023-12-31_12-05-57/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.04, 0.99]-bart-softmax_continous/2023-12-31_12-05-57/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.7]-bart-softmax_continous/2024-01-22_13-07-17/checkpoints/last.ckpt"
# 0.32
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.16
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.08
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005

# curriculum reverse resume from ckpt, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-softmax/2024-01-24_20-00-58/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-softmax/2024-01-24_20-01-00/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-softmax/2024-01-25_04-57-52/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-softmax/2024-01-24_20-01-00/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=64
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.32, 0.99]-bart-vqvae/2024-01-16_14-25-28/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.16, 0.99]-bart-vqvae/2024-01-16_06-54-27/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-16_06-51-30/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-16_14-22-28/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.7]-bart-vqvae/2024-01-22_13-10-16/checkpoints/last.ckpt"
# 0.32
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.7] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.16
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.7] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.7] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.7] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=64
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-gumbel/2024-01-02_13-31-52/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.16, 0.99]-bart-gumbel/2024-01-02_13-31-53/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.08, 0.99]-bart-gumbel/2024-01-02_13-31-52/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.04, 0.99]-bart-gumbel/2024-01-02_13-31-54/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-18_07-48-15/checkpoints/model-12336-22.3472.ckpt"
# 0.32
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.16
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001


# curriculum reverse resume from ckpt, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-gumbel/2024-01-20_10-58-08/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-gumbel/2024-01-20_10-58-09/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-20_10-58-09/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-20_10-58-10/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.32 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

# curriculum reverse resume from ckpt again, 1 gpu:
# 0.32
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-gumbel/2024-01-22_13-22-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-gumbel/2024-01-22_13-22-17/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-22_13-22-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-22_13-22-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
DEVICE=2
BSIZE=128
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-softmax_continous/2024-01-02_17-47-10/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001



# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-softmax_continous/2024-01-08_13-01-30/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-softmax_continous/2024-01-14_05-03-33/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.01, 0.99]-bart-softmax/2024-01-17_13-16-18/checkpoints/last.ckpt"
# 0.16
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.08
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.02
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=32
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-vqvae/2024-01-09_13-14-31/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-vqvae/2024-01-09_14-14-49/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-vqvae/2024-01-08_13-01-31/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-vqvae/2024-01-08_13-04-32/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001



# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-20_10-31-08/checkpoints/last.ckpt"
# 0.16
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.08
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.02
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# curriculum reverse resume from ckpt, 1 gpu:
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.16, 0.99]-bart-vqvae/2024-01-25_06-34-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.08, 0.99]-bart-vqvae/2024-01-25_06-34-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.04, 0.99]-bart-vqvae/2024-01-25_06-37-15/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.02, 0.99]-bart-vqvae/2024-01-25_06-40-17/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=128
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-gumbel/2024-01-02_17-59-15/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-gumbel/2024-01-02_18-05-19/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-gumbel/2024-01-02_18-05-20/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-gumbel/2024-01-02_18-32-39/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001



# curriculum reverse, 1 gpu:
# 0.16
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-gumbel/2024-01-09_08-09-33/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.08
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.02
# python3 run_train.py +experiment=cogs_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
DEVICE=2
BSIZE=128
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.08, 0.99]-bart-softmax_continous/2023-12-30_12-54-14/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.04, 0.99]-bart-softmax_continous/2023-12-30_12-54-14/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.02, 0.99]-bart-softmax_continous/2023-12-30_12-48-13/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.01
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-softmax_continous/2023-12-30_12-39-13/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005


# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-softmax/2024-01-20_09-58-06/checkpoints/last.ckpt"
# 0.08
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.02
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.01
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=64
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-18_17-48-37/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-19_05-53-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.02, 0.99]-bart-vqvae/2024-01-18_17-39-38/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.01
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-vqvae/2024-01-14_10-41-13/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001



# curriculum reverse, 1 gpu:
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-vqvae/2024-01-21_07-47-17/checkpoints/last.ckpt"
# 0.08
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.04
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.02
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001
# 0.01
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001

# curriculum reverse resume from ckpt, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.08, 0.99]-bart-vqvae/2024-01-25_06-43-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.4 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=10 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.04, 0.99]-bart-vqvae/2024-01-25_06-43-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.35 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=15 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.02, 0.99]-bart-vqvae/2024-01-25_06-43-18/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.01
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.01, 0.99]-bart-vqvae/2024-01-25_06-46-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.3 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=20 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

# -------------------------------------------------------------------------------------------------------------------- #
DEVICE=2
BSIZE=128
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# curriculum, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.08, 0.99]-bart-gumbel/2024-01-03_00-49-25/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.04, 0.99]-bart-gumbel/2024-01-03_00-43-23/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.02, 0.99]-bart-gumbel/2024-01-03_00-40-21/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001
# 0.01
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-gumbel/2024-01-03_00-28-17/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001



# curriculum reverse, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-gumbel/2024-01-09_13-05-29/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.04
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.02
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005
# 0.01
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005

# curriculum reverse resume from ckpt, 1 gpu:
# 0.08
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-14_18-28-01/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.04
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-14_18-28-01/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.02
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.02, 0.99]-bart-gumbel/2024-01-15_15-21-38/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0
# 0.01
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.01, 0.99]-bart-gumbel/2024-01-15_05-38-15/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum_reverse.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.5 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.5 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0

deactivate
module purge
