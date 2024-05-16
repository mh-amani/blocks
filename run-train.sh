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

# DEVICE=2
# +model.map_location='cuda:0'
BSIZE=512
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
DEVICE=[0]
NAME="scan_final"
LR=0.001
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=1000

# supervised:
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true

# mixed:
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
    model.model_params.usex=False model.model_params.usez=False \
    model.model_params.loss_coeff.zxz=1.0 model.model_params.loss_coeff.xzx=1.0 \
    model.model_params.loss_coeff.quantization_zxz=0.0 model.model_params.loss_coeff.quantization_xzx=0.0 \
    model.model_params.loss_coeff.supervised_seperated_x=0.0 model.model_params.loss_coeff.supervised_seperated_z=0.0 \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +model.model_params.average_eos_in_backprop=True\
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss/zxz" num_epochs=$NUM_EPOCHS logger.wandb.notes="zxz_nomask" logger.wandb.tags=["zxz_nomask"]

# continue training:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.02, 0.99]-bart-vqvae/2024-01-26_05-02-12/checkpoints/last.ckpt'"
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/mixed-[0.01, 0.99]-bart-gumbel/2024-01-24_15-47-58/checkpoints/model-46116-0.0978.ckpt'"

python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
    model.substitute_config.model_params.loss_coeff.zxz=1.0 model.substitute_config.model_params.loss_coeff.xzx=0.0 \
    model.substitute_config.model_params.loss_coeff.quantization_zxz=0.0 model.substitute_config.model_params.loss_coeff.quantization_xzx=0.0 \
    model.substitute_config.model_params.loss_coeff.supervised_seperated_x=1.0 model.substitute_config.model_params.loss_coeff.supervised_seperated_z=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.4 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.4 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss/zxz" num_epochs=$NUM_EPOCHS logger.wandb.notes="unsupsup  continueing" logger.wandb.tags=["unsupsup","continue"]


# testing
BSIZE=512
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'
DEVICE=[0]
NAME="scan_final"
LR=999
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=1000
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.08, 0.99]-bart-vqvae/2024-01-26_11-36-24/checkpoints/last.ckpt'"
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/mixed-[0.01, 0.99]-bart-gumbel/2024-01-24_15-47-58/checkpoints/model-46116-0.0978.ckpt'"
python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.1111,0.9999] \
    trainer.devices=$DEVICE training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE +model.map_location='cuda:0' \
    sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" model.optimizer.lr=$LR


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- sfst --------------------------------------------------------------- #
# # supervised:
# DEVICE=2
# model.map_location: 'cuda:0'
BSIZE=256
DISC='sofmax' # 'gumbel' or 'vqvae' or 'softmax'
DEVICE=[0]
NAME="sfst_final"
LR=0.0001
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=1000
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.04, 0.99]-bart-vqvae/2024-01-20_17-14-13/checkpoints/last.ckpt'"
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-21_12-04-12/checkpoints/model-1060-9.7591.ckpt'"

# supervised:
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true

# mixed:
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
    model.model_params.usex=False model.model_params.usez=False \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss/zxz" num_epochs=$NUM_EPOCHS logger.wandb.notes="only zxz" logger.wandb.tags=["zxz"]

# continue training:
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
    model.substitute_config.model_params.loss_coeff.zxz=1.0 model.substitute_config.model_params.loss_coeff.xzx=0.0 \
    model.substitute_config.model_params.loss_coeff.quantization_zxz=0.0 \
    model.substitute_config.model_params.loss_coeff.supervised_seperated_x=1.0 model.substitute_config.model_params.loss_coeff.supervised_seperated_z=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.0 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss/zxz" num_epochs=$NUM_EPOCHS logger.wandb.notes="zxz - continueing" logger.wandb.tags=["zxz"]


# testing
python3 run_inference.py +experiment/inference=inference datamodule=scan datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set ------------------------------------------------------------- #
# # supervised:
DEVICE=2
BSIZE=64
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

# mixed
python3 run_train.py +experiment=pcfgset_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp || true


# testing
python3 run_inference.py +experiment/inference=inference datamodule=pcfgset datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ----------------------------------------------------------------#
# use BPE tokenizer
# # supervised:
DEVICE=2
BSIZE=64
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

# mixed
python3 run_train.py +experiment=cogs_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp datamodule.dataset_parameters.num_workers=1 || true


# testing
python3 run_inference.py +experiment/inference=inference datamodule=cogs datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] trainer.devices=[$DEVICE] training_type=suponly datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" || true

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ---------------------------------------------------------------- #
# use BPE tokenizer
# # supervised:
DEVICE=2
BSIZE=64
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# supervised:
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.16,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.32,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=$DISC trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true

# weakly supervised:
# 4 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-17_15-44-19/checkpoints/last.ckpt'"
# 8 layer model:
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-21_14-02-58/checkpoints/last.ckpt'"
python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" +test=True
python3 run_train.py +experiment=cfq_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] trainer.devices=[$DEVICE] datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=gpt2_gpt2 discretizer_key=$DISC model.checkpoint_path="$CKPT" model/lr_scheduler=cosine_annealing +test=True

# mixed
python3 run_train.py +experiment=cfq_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True trainer=ddp datamodule.dataset_parameters.num_workers=1 || true


# testing


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# deactivate
# module purge

python3 run_train.py +experiment=cfq_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=vqvae trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 +test=True
python3 run_train.py +experiment=scan_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] datamodule.dataset_parameters.num_workers=1 model/discretizer=vqvae trainer.devices=[0] datamodule.dataset_parameters.batch_size=256 +test=True