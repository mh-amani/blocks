#  curr sup unsup

BSIZE=256
NAME="scan_final"
DEVICE=[0]
LR=0.0007
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=2000

# #########
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'

# # 3x7pgdta
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.01, 0.9]-bart-gumbel/2023-12-27_22-47-53/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]


# uj9qbqh7
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.02, 0.9]-bart-gumbel/2023-12-28_12-42-36/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=$DEVICE\
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]

# # b3fo0obs
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-bart-gumbel/2023-12-28_16-55-18/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]

# 7sqx9wg5
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.08, 0.9]-bart-gumbel/2023-12-28_21-17-19/checkpoints/last.ckpt'"
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]



