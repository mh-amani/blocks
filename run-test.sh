#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:40gb:1
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=./slurm_out_test/test_sym_ae_%j.out
#SBATCH --error=./slurm_err_test/test_sym_ae_%j.err

module load miniconda/3
conda activate blocks

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
# export WANDB_MODE=offline

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set ------------------------------------------------------------- #

# ---------------------------------------------- Softmax ------------------------------------------------------------- #
BSIZE=512
DISC='softmax'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.7]-bart-softmax_continous/2024-01-22_13-07-17/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# supervised:

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.04, 0.99]-bart-softmax_continous/2023-12-31_12-05-57/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.08, 0.99]-bart-softmax_continous/2023-12-31_12-05-57/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.16, 0.99]-bart-softmax_continous/2023-12-31_12-05-56/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# 0.32
# training_type="suponly-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-softmax_continous/2023-12-31_12-08-58/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.99, 0.99]-bart-softmax_continous/2023-12-31_12-54-13/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.04, 0.99]-bart-softmax_continous/2024-01-02_13-10-50/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.08, 0.99]-bart-softmax_continous/2024-01-02_13-07-49/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.16, 0.99]-bart-softmax_continous/2023-12-30_12-57-16/checkpoints/last.ckpt'"

# 0.32
# training_type="mixed-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.32, 0.99]-bart-softmax_continous/2023-12-30_12-57-17/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr sup --> unsupervised
# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.04, 0.99]-bart-softmax/2024-01-14_14-12-16/checkpoints/last.ckpt'"
# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.08, 0.99]-bart-softmax/2024-01-14_19-07-04/checkpoints/last.ckpt'"
# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.16, 0.99]-bart-softmax/2024-01-14_13-48-09/checkpoints/last.ckpt'"
# 0.32
# training_type="curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.32, 0.99]-bart-softmax/2024-01-14_13-48-10/checkpoints/last.ckpt'"

# curr unsup -> sup 

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.04, 0.99]-bart-softmax/2024-01-27_18-30-20/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-softmax/2024-01-27_17-36-06/checkpoints/last.ckpt'"

# 0.16
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-softmax/2024-01-27_16-38-54/checkpoints/last.ckpt'"

# 0.32
# training_type="reverse-curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-softmax/2024-01-27_09-48-09/checkpoints/last.ckpt'"


# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# ---------------------------------------------- VQVAE ------------------------------------------------------------- #
BSIZE=512
DISC='vqvae'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.01, 0.99]-bart-vqvae/2024-01-14_10-29-12/checkpoints/last.ckpt'"

# supervised:
# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-16_14-22-28/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-16_06-51-30/checkpoints/last.ckpt'"

# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.16, 0.99]-bart-vqvae/2024-01-16_06-54-27/checkpoints/last.ckpt'"

# 0.32
# training_type="suponly-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.32, 0.99]-bart-vqvae/2024-01-16_14-25-28/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.99, 0.99]-bart-vqvae/2024-01-08_12-25-19/checkpoints/last.ckpt'"


# mixed:
# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.04, 0.99]-bart-vqvae/2024-01-03_20-29-59/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.08, 0.99]-bart-vqvae/2024-01-03_20-29-59/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.16, 0.99]-bart-vqvae/2024-01-03_22-00-21/checkpoints/last.ckpt'"

# 0.32
# training_type="mixed-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.32, 0.99]-bart-vqvae/2024-01-03_22-00-21/checkpoints/last.ckpt'"

# curr sup --> unsupervised
# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-18_17-27-37/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-18_10-15-37/checkpoints/last.ckpt'"

# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.16, 0.99]-bart-vqvae/2024-01-18_07-54-18/checkpoints/last.ckpt'"

# 0.32
# training_type="curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.32, 0.99]-bart-vqvae/2024-01-18_07-54-18/checkpoints/last.ckpt'"


# curr unsup -> sup
# 0.04* not great
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.04, 0.99]-bart-vqvae/2024-01-25_06-07-09/checkpoints/last.ckpt'"

# 0.08*
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-vqvae/2024-01-25_06-07-10/checkpoints/last.ckpt'"

# 0.16*
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-vqvae/2024-01-25_06-04-07/checkpoints/last.ckpt'"

# 0.32*
# training_type="reverse-curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-vqvae/2024-01-25_20-24-26/checkpoints/last.ckpt'"


# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# ---------------------------------------------- Gumbel ------------------------------------------------------------- #
BSIZE=512
DISC='gumbel'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-18_07-48-15/checkpoints/model-12336-22.3472.ckpt'"

# supervised:
# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.04, 0.99]-bart-gumbel/2024-01-02_13-31-54/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.08, 0.99]-bart-gumbel/2024-01-02_13-31-52/checkpoints/last.ckpt'"

# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.16, 0.99]-bart-gumbel/2024-01-02_13-31-53/checkpoints/last.ckpt'"

# 0.32
# training_type="suponly-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.32, 0.99]-bart-gumbel/2024-01-02_13-31-52/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/suponly-[0.99, 0.99]-bart-gumbel/2024-01-02_14-05-14/checkpoints/last.ckpt'"


# mixed:
# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.04, 0.99]-bart-gumbel/2024-01-03_23-06-42/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.08, 0.99]-bart-gumbel/2024-01-03_23-03-41/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.16, 0.99]-bart-gumbel/2024-01-03_23-03-41/checkpoints/last.ckpt'"

# 0.32
# training_type="mixed-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/mixed-[0.32, 0.99]-bart-gumbel/2024-01-03_23-03-39/checkpoints/last.ckpt'"


# curr sup --> unsupervised
# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.04, 0.99]-bart-gumbel/2024-01-14_16-30-38/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.08, 0.99]-bart-gumbel/2024-01-14_16-27-35/checkpoints/last.ckpt'"

# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.16, 0.99]-bart-gumbel/2024-01-14_16-21-34/checkpoints/last.ckpt'"

# 0.32
# training_type="curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-[0.32, 0.99]-bart-gumbel/2024-01-14_16-21-33/checkpoints/last.ckpt'"


# curr unsup -> sup
# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-25_05-43-04/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-25_05-31-01/checkpoints/last.ckpt'"

# 0.16
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.16, 0.99]-bart-gumbel/2024-01-25_05-24-59/checkpoints/last.ckpt'"

# 0.32
# training_type="reverse-curr-0.32"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/pcfg_set/curriculum-reverse-[0.32, 0.99]-bart-gumbel/2024-01-25_05-12-55/checkpoints/last.ckpt'"


# python3 run_inference.py +experiment/inference=inference_pcfg datamodule=pcfg_set training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#

# ---------------------------------------------- Softmax ------------------------------------------------------------- #
BSIZE=128
DISC='softmax'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.01, 0.99]-bart-softmax/2024-01-17_13-16-18/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# supervised:

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-softmax_continous/2024-01-02_17-47-10/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt'"

# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-softmax_continous/2024-01-02_17-44-09/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.99, 0.99]-bart-softmax_continous/2024-01-02_17-44-08/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:


# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.02, 0.99]-bart-softmax_continous/2024-01-02_17-47-10/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.04, 0.99]-bart-softmax_continous/2024-01-02_17-47-11/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.08, 0.99]-bart-softmax_continous/2024-01-02_17-47-11/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.16, 0.99]-bart-softmax_continous/2024-01-02_17-47-08/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.02, 0.99]-bart-softmax/2024-01-14_14-39-21/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.04, 0.99]-bart-softmax/2024-01-14_14-39-20/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.08, 0.99]-bart-softmax/2024-01-14_14-33-19/checkpoints/last.ckpt'"

# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.16, 0.99]-bart-softmax/2024-01-14_19-25-06/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup 


# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.02, 0.99]-bart-softmax/2024-01-20_10-16-09/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.04, 0.99]-bart-softmax/2024-01-20_10-16-08/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.08, 0.99]-bart-softmax/2024-01-20_10-16-08/checkpoints/last.ckpt'"

# 0.16
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.16, 0.99]-bart-softmax/2024-01-20_10-16-06/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# ---------------------------------------------- VQVAE ------------------------------------------------------------- #
BSIZE=32
DISC='vqvae'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-25_06-40-18/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# supervised:

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-vqvae/2024-01-08_13-04-32/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-vqvae/2024-01-08_13-01-31/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-vqvae/2024-01-13_13-14-57/checkpoints/last.ckpt'"

# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-vqvae/2024-01-09_13-14-31/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.99, 0.99]-bart-vqvae/2024-01-08_13-04-31/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.02, 0.99]-bart-vqvae/2024-01-08_13-10-33/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.04, 0.99]-bart-vqvae/2024-01-08_13-10-33/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.08, 0.99]-bart-vqvae/2024-01-08_13-10-32/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.16, 0.99]-bart-vqvae/2024-01-08_13-13-32/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.02, 0.99]-bart-vqvae/2024-01-14_16-12-31/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-14_16-12-30/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-14_16-12-30/checkpoints/last.ckpt'"

# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.16, 0.99]-bart-vqvae/2024-01-14_16-12-30/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup

# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.02, 0.99]-bart-vqvae/2024-01-27_11-24-40/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.04, 0.99]-bart-vqvae/2024-01-27_11-24-40/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.08, 0.99]-bart-vqvae/2024-01-27_11-24-40/checkpoints/last.ckpt'"

# 0.16
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.16, 0.99]-bart-vqvae/2024-01-27_11-24-40/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# ---------------------------------------------- Gumbel ------------------------------------------------------------- #
BSIZE=512
DISC='gumbel'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.01, 0.99]-bart-gumbel/2024-01-09_08-09-33/checkpoints/last.ckpt'"
# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# supervised:

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.02, 0.99]-bart-gumbel/2024-01-02_18-32-39/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.04, 0.99]-bart-gumbel/2024-01-02_18-05-20/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.08, 0.99]-bart-gumbel/2024-01-02_18-05-19/checkpoints/last.ckpt'"

# 0.16
# training_type="suponly-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.16, 0.99]-bart-gumbel/2024-01-02_17-59-15/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/suponly-[0.99, 0.99]-bart-gumbel/2024-01-02_17-59-17/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.02, 0.99]-bart-gumbel/2024-01-08_18-28-45/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.04, 0.99]-bart-gumbel/2024-01-08_21-42-41/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.08, 0.99]-bart-gumbel/2024-01-08_22-40-13/checkpoints/last.ckpt'"

# 0.16
# training_type="mixed-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/mixed-[0.16, 0.99]-bart-gumbel/2024-01-09_05-26-49/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.02, 0.99]-bart-gumbel/2024-01-14_18-03-59/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.04, 0.99]-bart-gumbel/2024-01-14_18-03-59/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.08, 0.99]-bart-gumbel/2024-01-14_17-58-00/checkpoints/last.ckpt'"

# 0.16
# training_type="curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-[0.16, 0.99]-bart-gumbel/2024-01-14_16-33-36/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_suponly datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup

# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.02, 0.99]-bart-gumbel/2024-01-16_06-24-24/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-16_05-39-20/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-16_05-39-20/checkpoints/last.ckpt'"

# 0.16
# training_type="reverse-curr-0.16"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cogs/curriculum-reverse-[0.16, 0.99]-bart-gumbel/2024-01-16_05-27-19/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cogs_curriculum datamodule=cogs training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #

# ---------------------------------------------- Softmax ------------------------------------------------------------- #
BSIZE=512
DISC='softmax'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1


# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-softmax/2024-01-20_09-58-06/checkpoints/last.ckpt'"

# supervised:

# 0.01
# training_type="suponly-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-softmax_continous/2023-12-30_12-39-13/checkpoints/last.ckpt'"

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.02, 0.99]-bart-softmax_continous/2023-12-30_12-48-13/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.04, 0.99]-bart-softmax_continous/2023-12-30_12-54-14/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.08, 0.99]-bart-softmax_continous/2023-12-30_12-54-14/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.99, 0.99]-bart-softmax_continous/2023-12-31_13-06-17/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.01
# training_type="mixed-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-softmax_continous/2023-12-30_12-57-13/checkpoints/last.ckpt'"

# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.02, 0.99]-bart-softmax_continous/2023-12-30_12-57-17/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.04, 0.99]-bart-softmax_continous/2023-12-30_13-03-15/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.08, 0.99]-bart-softmax_continous/2023-12-30_13-09-15/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.01
# training_type="curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-softmax/2024-01-14_15-30-24/checkpoints/last.ckpt'"

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.02, 0.99]-bart-softmax/2024-01-14_15-27-26/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.04, 0.99]-bart-softmax/2024-01-14_15-27-26/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.08, 0.99]-bart-softmax/2024-01-14_15-24-30/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup 

# 0.01
# training_type="reverse-curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.01, 0.99]-bart-softmax/2024-01-21_08-05-18/checkpoints/last.ckpt'"

# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.02, 0.99]-bart-softmax/2024-01-21_08-02-18/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.04, 0.99]-bart-softmax/2024-01-21_08-02-16/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.08, 0.99]-bart-softmax/2024-01-21_08-02-18/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# ---------------------------------------------- VQVAE ------------------------------------------------------------- #
BSIZE=32
DISC='vqvae'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-vqvae/2024-01-21_07-47-17/checkpoints/last.ckpt'"

# supervised:

# 0.01
# training_type="suponly-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-vqvae/2024-01-14_10-41-13/checkpoints/last.ckpt'"

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.02, 0.99]-bart-vqvae/2024-01-18_17-39-38/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-19_05-53-18/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-18_17-48-37/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.99, 0.99]-bart-vqvae/2024-01-14_11-35-27/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.01
# training_type="mixed-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-vqvae/2024-01-09_09-57-55/checkpoints/last.ckpt'"

# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.02, 0.99]-bart-vqvae/2024-01-09_10-03-57/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.04, 0.99]-bart-vqvae/2024-01-09_14-20-51/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.08, 0.99]-bart-vqvae/2024-01-09_10-22-02/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.01
# training_type="curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-20_10-46-07/checkpoints/last.ckpt'"

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.02, 0.99]-bart-vqvae/2024-01-20_10-46-07/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-20_10-46-07/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-20_10-46-09/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup 

# 0.01
# training_type="reverse-curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.01, 0.99]-bart-vqvae/2024-01-27_22-23-18/checkpoints/last.ckpt'"

# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.02, 0.99]-bart-vqvae/2024-01-27_22-11-13/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.04, 0.99]-bart-vqvae/2024-01-27_22-17-13/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.08, 0.99]-bart-vqvae/2024-01-27_21-50-06/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]
# ---------------------------------------------- Gumbel ------------------------------------------------------------- #
BSIZE=512
DISC='gumbel'
LR=999
SEQMODEL='bart'
NUM_EPOCHS=1000
NUM_WORKERS=1

# unsupervised:
# 0.01
# training_type="unsup-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-18_07-48-15/checkpoints/last.ckpt'"

# supervised:

# 0.01
# training_type="suponly-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.01, 0.99]-bart-gumbel/2024-01-03_00-28-17/checkpoints/last.ckpt'"

# 0.02
# training_type="suponly-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.02, 0.99]-bart-gumbel/2024-01-03_00-40-21/checkpoints/last.ckpt'"

# 0.04
# training_type="suponly-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.04, 0.99]-bart-gumbel/2024-01-03_00-43-23/checkpoints/last.ckpt'"

# 0.08
# training_type="suponly-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.08, 0.99]-bart-gumbel/2024-01-03_00-49-25/checkpoints/last.ckpt'"

# 0.99
# training_type="suponly-0.99"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/suponly-[0.99, 0.99]-bart-gumbel/2024-01-03_00-55-28/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# mixed:

# 0.01
# training_type="mixed-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.01, 0.99]-bart-gumbel/2024-01-03_01-37-40/checkpoints/last.ckpt'"

# 0.02
# training_type="mixed-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.02, 0.99]-bart-gumbel/2024-01-03_01-31-38/checkpoints/last.ckpt'"

# 0.04
# training_type="mixed-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.04, 0.99]-bart-gumbel/2024-01-03_01-28-38/checkpoints/last.ckpt'"

# 0.08
# training_type="mixed-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/mixed-[0.08, 0.99]-bart-gumbel/2024-01-03_01-25-37/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]


# curr sup --> unsupervised

# 0.01
# training_type="curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-14_18-27-59/checkpoints/last.ckpt'"

# 0.02
# training_type="curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.02, 0.99]-bart-gumbel/2024-01-14_18-19-00/checkpoints/last.ckpt'"

# 0.04
# training_type="curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.04, 0.99]-bart-gumbel/2024-01-14_18-13-00/checkpoints/last.ckpt'"

# 0.08
# training_type="curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-[0.08, 0.99]-bart-gumbel/2024-01-14_18-13-00/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

# curr unsup -> sup 

# 0.01
# training_type="reverse-curr-0.01"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.01, 0.99]-bart-gumbel/2024-01-20_11-16-10/checkpoints/last.ckpt'"

# 0.02
# training_type="reverse-curr-0.02"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.02, 0.99]-bart-gumbel/2024-01-20_11-16-10/checkpoints/last.ckpt'"

# 0.04
# training_type="reverse-curr-0.04"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.04, 0.99]-bart-gumbel/2024-01-20_11-16-10/checkpoints/last.ckpt'"

# 0.08
# training_type="reverse-curr-0.08"
# CKPT="'/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/cfq/curriculum-reverse-[0.08, 0.99]-bart-gumbel/2024-01-20_11-16-10/checkpoints/last.ckpt'"

# python3 run_inference.py +experiment/inference=inference_cfq datamodule=cfq training_type=$training_type trainer.devices=[0] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR datamodule.dataset_parameters.num_workers=$NUM_WORKERS logger.wandb.tags=["use-val"]

deactivate
module purge
