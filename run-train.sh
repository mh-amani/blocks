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
# export WANDB_MODE=offline
# export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef

# source /dlabdata1/masani/miniconda3/bin/activate

# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN --------------------------------------------------------------- #

DEVICE=2
BSIZE=256
DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'
DEVICE=[0]
NAME="scan_final"
LR=0.001
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'

# supervised, 02
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.02, 0.9]-bart-softmax_continous/2023-12-28_12-38-04/checkpoints/last.ckpt'"

# supervised, 04

# unsupervised
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.02, 0.9]-bart-softmax/2023-12-29_21-02-35/checkpoints/last.ckpt'"

# supervised:
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
# python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
# python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true

# continue training
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True model.optimizer=None model.substitute_config=None logger.wandb.notes="only zxz" name=$NAME 

# zxz-only to supervised
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/mixed-[0.01, 0.99]-bart-softmax_continous/2023-12-30_15-53-25/checkpoints/last.ckpt'"
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True model.optimizer.lr=$LR logger.wandb.notes="unsup to sup" name=$NAME model.substitute_config.model_params.usex=False callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.2 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 


# weakly supervised:
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True model.optimizer.lr=$LR logger.wandb.notes="only zxz" name=$NAME model.substitute_config.model_params.usex=True
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True

# otherwise mixed unsup training:
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True logger.wandb.notes="mixed xz -> 0"

# only zxz
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 model.model_params.loss_coeff.zxz=1.0 model.lr_scheduler.monitor="val/loss/zxz" model.optimizer.lr=$LR num_epochs=1000 logger.wandb.notes="only zxz" name=$NAME 

# xzx and zxz   
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 callbacks.supervision_scheduler.scheduler_z.hp_init=0.6 callbacks.supervision_scheduler.scheduler_z.hp_end=0.6 model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 model.lr_scheduler.monitor="val/loss" model.optimizer.lr=$LR num_epochs=1000 logger.wandb.notes="only zxz and xzx"
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/mixed-[0.01, 0.7]-bart-softmax_continous/2024-01-03_10-11-23/checkpoints/last.ckpt'"


# testing
python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- sfst --------------------------------------------------------------- #
# # supervised:
DEVICE=0
BSIZE=256
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'


# supervised:
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true

# weakly supervised:
# 4 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-17_15-44-19/checkpoints/last.ckpt'"
# 8 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-21_14-02-58/checkpoints/last.ckpt'"
python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True


# testing
python3 run_inference.py +experiment/inference=inference datamodule=sfst datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set ------------------------------------------------------------- #
# # supervised:
DEVICE=0
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised:
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true

# weakly supervised:
# 4 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-17_15-44-19/checkpoints/last.ckpt'"
# 8 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-21_14-02-58/checkpoints/last.ckpt'"
python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=pcfgset_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True


# testing
python3 run_inference.py +experiment/inference=inference datamodule=pcfgset datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# # supervised:
DEVICE=0
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised:
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true

# weakly supervised:
# 4 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-17_15-44-19/checkpoints/last.ckpt'"
# 8 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-21_14-02-58/checkpoints/last.ckpt'"
python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=cogs_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True


# testing
python3 run_inference.py +experiment/inference=inference datamodule=cogs datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# # supervised:
DEVICE=0
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised:
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true

# weakly supervised:
# 4 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-17_15-44-19/checkpoints/last.ckpt'"
# 8 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-21_14-02-58/checkpoints/last.ckpt'"
python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True


# testing
python3 run_inference.py +experiment/inference=inference datamodule=cfq datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# deactivate
# module purge
