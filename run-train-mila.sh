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
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'

# supervised
# 1 gpu
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true
# continue from ckpt
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.04, 0.99]-bart-vqvae/2024-01-14_05-06-34/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.08, 0.99]-bart-vqvae/2024-01-14_05-06-34/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.16, 0.99]-bart-vqvae/2024-01-14_05-06-33/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-vqvae/2024-01-14_05-09-35/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true

# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 || true

# only zxz
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE model.optimizer.lr=0.001 +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# from ckpt
# softmax
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.99]-bart-softmax_continous/2024-01-14_13-48-09/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.01, 0.99]-bart-softmax/2024-01-17_13-16-17/checkpoints/last.ckpt"
# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.99]-bart-vqvae/2024-01-14_10-29-12/checkpoints/last.ckpt"
# gumbel
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.99]-bart-gumbel/2024-01-14_16-33-36/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.lr_scheduler.monitor='val/loss/zxz' logger.wandb.notes="only zxz" || true

# zxz + xzx
# python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.7 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 \
#     +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx"

# from ckpt
# softmax
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.7]-bart-softmax_continous/2024-01-22_13-07-17/checkpoints/last.ckpt"
# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.7]-bart-vqvae/2024-01-22_13-10-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.9 callbacks.supervision_scheduler.scheduler_z.hp_end=0.9 \
#     +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx" "model.checkpoint_path='$CKPT'"

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=32
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'
# supervised
# 1 gpu
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true

# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp  model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true


# mixed
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.002 || true
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001 model.model_params.usez=True model.model_params.loss_coeff.zxz=0.1 || true
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.90, 0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.001,0.0005 model.model_params.usex=True model.model_params.loss_coeff.xzx=0.1 model.model_params.usez=True model.model_params.loss_coeff.zxz=0.1 || true


# only zxz
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' model.optimizer.lr=0.0005 || true

# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-vqvae/2024-01-08_13-13-32/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-20_10-31-08/checkpoints/last.ckpt"
# softmax
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-softmax_continous/2024-01-14_05-03-33/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0001 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.lr_scheduler.monitor='val/loss/zxz' logger.wandb.notes="only zxz" model.substitute_config.model_params.max_z_length=180 || true

# zxz + xzx
# python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.7 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 \
#     +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx"

# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.7]-bart-vqvae/2024-01-22_13-34-17/checkpoints/last.ckpt"
# from ckpt again
# python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.7 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 \
#     +test=True model.optimizer.lr=0.0005 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx" "model.checkpoint_path='$CKPT'" model.substitute_config.model_params.max_z_length=180


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# supervised:
DEVICE=2
BSIZE=64
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
SEQMODEL='bart'

# supervised
# 1 gpu
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.001 || true
# 1 gpu, val_loss_separated for lr scheduler
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True model.optimizer.lr=0.01 model.lr_scheduler.monitor='val/loss/supervised_seperated' || true

# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True datamodule.dataset_parameters.num_workers=1 model.optimizer.lr=0.001 || true

# continue from ckpt
# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-vqvae/2024-01-14_10-41-13/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.02, 0.99]-bart-vqvae/2024-01-14_10-41-14/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.04, 0.99]-bart-vqvae/2024-01-14_10-41-14/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.08, 0.99]-bart-vqvae/2024-01-14_10-41-14/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.005 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.99, 0.99]-bart-vqvae/2024-01-14_11-35-27/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.99] discretizer_key=$DISC "model.checkpoint_path='$CKPT'" trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL +test=True model.optimizer.lr=0.01 model.lr_scheduler.monitor='val/loss/supervised_seperated' callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 || true



# mixed
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp model.optimizer.lr=0.001 || true
# mixed, 1 gpu
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.002 || true
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.90] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed" model.optimizer.lr=0.0005 model.model_params.usex=True model.model_params.loss_coeff.xzx=0.1 || true

# only zxz
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' || true
# only zxz, 1 gpu
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="only zxz" callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor='val/loss/zxz' model.optimizer.lr=0.001 || true

# only zxz, 1 gpu, load from ckpt
# vqvae
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-vqvae/2024-01-09_10-25-03/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-14_12-29-42/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-vqvae/2024-01-21_07-47-17/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.lr_scheduler.monitor='val/loss/zxz' logger.wandb.notes="only zxz" || true
# softmax
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-softmax_continous/2024-01-16_06-48-27/checkpoints/last.ckpt"
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-softmax/2024-01-17_13-16-18/checkpoints/last.ckpt"
# gumbel
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-gumbel/2024-01-09_13-05-29/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC "model.checkpoint_path='$CKPT'" +test=True model.optimizer.lr=0.0005 callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.lr_scheduler.monitor='val/loss/zxz' logger.wandb.notes="only zxz" || true

# zxz + xzx
# python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.7 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 \
#     +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx"

# zxz + xzx load from ckpt
# CKPT="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.7]-bart-vqvae/2024-01-22_13-55-16/checkpoints/last.ckpt"
# python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] trainer.devices=[0] \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
#     model.model_params.loss_coeff.quantization_zxz=0.0 \
#     model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=0.7 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 \
#     +test=True model.optimizer.lr=0.001 model.lr_scheduler.monitor="val/loss/zxz" logger.wandb.notes="zxz and xzx"


deactivate
module purge
