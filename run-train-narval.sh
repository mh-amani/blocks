#!/bin/bash

#SBATCH --account=def-gdumas85
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=05:59:00
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
# wandb login

# export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/blocks/hack.so
export WANDB_MODE=offline

# model.collator.tokenizer.vocab_size, model.lr_scheduler.patience/cooldown, model.optimizer.lr

# for runs more than a day, use: 1-11:59:00 (day-hour)
# lists can be passed both as a string or as a list. Example: supervision_ratio=\[1.,0.0,0.0\] or 'supervision_ratio=[1.,0.0,0.0]'

# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- SCAN ---------------------------------------------- #

# ------------------------------------- Stage I - Fully Supervised ---------------------------------- #
# python run_train.py +experiment=scan_gpt2-gpt2_gumbel_supervised.yaml trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.8 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["scan","fully-supervised"] datamodule.dataset_parameters.num_workers=1


# ---------------------------- Stage II - Weakly Supervised (load from ckpt) ------------------------ #
# python run_train.py +experiment=scan_pretrained_weaksup.yaml run_name="supervised-curriculum-0.02-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.02, 0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.02-gumbel/2023-09-20_19-38-25/checkpoints/last.ckpt"
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------- PCFG Set -------------------------------------------- #

# ------------------------------------- Stage I - Fully Supervised ---------------------------------- #
# pcfgset
# 0.02
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.02-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 ++datamodule/experiment="pcfgset" datamodule.dataset_parameters.supervision_ratio="[0.02,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.04
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.04-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=512 model.optimizer.lr=0.001 ++datamodule/experiment="pcfgset" datamodule.dataset_parameters.supervision_ratio="[0.04,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.06
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.06-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 ++datamodule/experiment="pcfgset" datamodule.dataset_parameters.supervision_ratio="[0.06,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.08
# python run_train.py +experiment=pcfgset_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.08-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 ++datamodule/experiment="pcfgset" datamodule.dataset_parameters.supervision_ratio="[0.08,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100

# ---------------------------- Stage II - Weakly Supervised (load from ckpt) ------------------------ #

# pcfgset
# 0.02
# python run_train.py +experiment=pcfgset_pretrained_weaksup run_name="supervised-curriculum-0.02-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 ++datamodule/experiment="pcfgset" model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.02,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.02-gumbel/2023-09-22_04-44-02/checkpoints/last.ckpt"
# 0.04
# python run_train.py +experiment=pcfgset_pretrained_weaksup run_name="supervised-curriculum-0.04-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 ++datamodule/experiment="pcfgset" model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.04,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.04-gumbel/2023-09-22_04-44-22/checkpoints/last.ckpt"
# 0.06
# python run_train.py +experiment=pcfgset_pretrained_weaksup run_name="supervised-curriculum-0.06-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 ++datamodule/experiment="pcfgset" model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.06,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.06-gumbel/2023-09-22_04-46-49/checkpoints/last.ckpt"
# 0.08
python run_train.py +experiment=pcfgset_pretrained_weaksup run_name="supervised-curriculum-0.08-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 ++datamodule/experiment="pcfgset" model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.08,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.08-gumbel/2023-09-22_04-52-52/checkpoints/last.ckpt"
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- COGS ---------------------------------------------- #

# ------------------------------------- Stage I - Fully Supervised ---------------------------------- #
# 0.002
# python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.002-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.002,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab-frac"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.004
# python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.004-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.004,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab-frac"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.006
# python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.006-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.006,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab-frac"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100
# 0.008
# python run_train.py +experiment=cogs_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.008-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=1024 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.008,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab-frac"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=100

# ---------------------------- Stage II - Weakly Supervised (load from ckpt) ------------------------ #
# 0.002
# python run_train.py +experiment=cogs_pretrained_weaksup run_name="supervised-curriculum-0.002-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.002,0.6]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.002-gumbel/2023-09-22_06-20-33/checkpoints/last.ckpt"
# 0.004
# python run_train.py +experiment=cogs_pretrained_weaksup run_name="supervised-curriculum-0.004-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.002,0.6]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.004-gumbel/2023-09-22_06-25-31/checkpoints/last.ckpt"
# 0.006
# python run_train.py +experiment=cogs_pretrained_weaksup run_name="supervised-curriculum-0.006-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.002,0.6]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.006-gumbel/2023-09-22_06-25-31/checkpoints/last.ckpt"
# 0.008
# python run_train.py +experiment=cogs_pretrained_weaksup run_name="supervised-curriculum-0.008-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.002,0.6]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.5 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.008-gumbel/2023-09-22_06-27-52/checkpoints/last.ckpt"
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- CFQ ----------------------------------------------- #

# ------------------------------------- Stage I - Fully Supervised ---------------------------------- #
# 0.002
# python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.002-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.002,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=200
# 0.004
# python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.004-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.004,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=200
# 0.006
# python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.006-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.006,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=200
# 0.008
python run_train.py +experiment=cfq_gpt2-gpt2_gumbel_supervised run_name="supervised-only-0.008-gumbel" trainer.accelerator='gpu' trainer.devices=1 trainer.min_epochs=100 datamodule.dataset_parameters.batch_size=256 model.optimizer.lr=0.001 datamodule.dataset_parameters.supervision_ratio="[0.008,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=1.0 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 logger.wandb.tags=["supervised-training","smaller-vocab"] datamodule.dataset_parameters.num_workers=1 model.collator.tokenizer.vocab_size=200

# ---------------------------- Stage II - Weakly Supervised (load from ckpt) ------------------------ #
# 0.002, vocab=1000
# python run_train.py +experiment=cfq_pretrained_weaksup run_name="supervised-curriculum-0.002-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.002,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.002-gumbel/2023-09-22_05-06-58/checkpoints/last.ckpt"
# 0.004, vocab=1000
# python run_train.py +experiment=cfq_pretrained_weaksup run_name="supervised-curriculum-0.004-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.004,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path=???
# 0.006, vocab=1000
# python run_train.py +experiment=cfq_pretrained_weaksup run_name="supervised-curriculum-0.006-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.006,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path=???
# 0.008, vocab=1000
# python run_train.py +experiment=cfq_pretrained_weaksup run_name="supervised-curriculum-0.008-gumbel" trainer.accelerator='gpu' trainer.devices=1 model.substitute_config.optimizer.lr=0.001 datamodule.dataset_parameters.batch_size=64 model.substitute_config.model_params.acc_grad_batch=1 datamodule.dataset_parameters.supervision_ratio="[0.008,0.9]" callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.7 callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=0.7 logger.wandb.tags=["weakly-supervised"] datamodule.dataset_parameters.num_workers=1 model.checkpoint_path="/home/aminm/blocks/logs/training/runs/supervised-only-0.008-gumbel/2023-09-22_05-07-53/checkpoints/last.ckpt"


deactivate
module purge
