#  curr sup unsup
# ckpt , disc
BSIZE=256
NAME="scan_final"
LR=0.0007
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=2000
DEVICE=[0]

# #########
DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

# # pefzv9mj
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.01, 0.9]-bart-vqvae/2023-12-27_14-26-18/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]


# m6m48imc
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.02, 0.9]-bart-vqvae/2023-12-27_18-20-46/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=$DEVICE\
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]

# # pe6j7xyk
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-bart-vqvae/2023-12-27_22-13-22/checkpoints/last.ckpt'"
# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]

# ofcdqhrn
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.08, 0.9]-bart-vqvae/2023-12-28_01-46-14/checkpoints/last.ckpt'"
python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False +model.map_location='cuda:0' \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=400 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]


