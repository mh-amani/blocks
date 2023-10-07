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


# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN ---------------------------------------------- #
# supervised:
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true

# weakly supervised:
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_17-32-17/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.02,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_18-54-07/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.04,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_20-31-35/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.08,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_21-33-06/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.16,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_22-08-55/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.32,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_22-44-20/checkpoints/last.ckpt\' +test=True || true

# testing
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/model-4488-3.0010.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true
# python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] trainer.devices=[1] training_type=suponly sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-03_11-48-13/checkpoints/last.ckpt\' || true




# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- sfst ---------------------------------------------- #
# # supervised:
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true
# python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=gumbel trainer.devices=[0] model.optimizer.lr=0.005 +test=True || true


# # weakly supervised:
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.01,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_10-52-30/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.02,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_13-27-15/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.04,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_15-24-53/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] trainer.devices=[0] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.08,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_17-16-20/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] trainer.devices=[1] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.16,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_19-40-12/checkpoints/last.ckpt\' +test=True || true
# python3 run_train.py +experiment=sfst_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] trainer.devices=[1] datamodule.dataset_parameters.batch_size=128 sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=gumbel model.checkpoint_path=\'/dlabdata1/masani/blocks/logs/training/runs/sfst/suponly-\[0.32,\ 0.9\]-gpt2_gpt2-gumbel/2023-10-04_21-34-51/checkpoints/last.ckpt\' +test=True || true

# # testing
# python3 run_inference.py +experiment/inference=scan_gpt2-gpt2_gumbel

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set -------------------------------------------- #
# supervised:
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true
# python3 run_train.py +experiment=pcfgset_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.9] model/discretizer=gumbel trainer.devices=[0] +test=True || true


# weakly supervised:
# python3 run_train.py +experiment=s




# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ---------------------------------------------- #
# use BPE tokenizer
# supervised:
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true
# python3 run_train.py +experiment=cogs_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.64,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true



# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ----------------------------------------------- #
# use BPE tokenizer
# supervised:
# python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] datamodule.dataset_parameters.batch_size=128 model/discretizer=gumbel trainer.devices=[1] +test=True || true


# deactivate
# module purge
