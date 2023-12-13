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
BSIZE=128
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
DEVICE=[1]
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-12-11_10-53-50/checkpoints/last.ckpt'"

# supervised:
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true


# weakly supervised:
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.7] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True

# only zxz, or otherwise mixed unsup training:
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp || true

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
