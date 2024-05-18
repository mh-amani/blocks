#  69qd7gix continued @ 6zoeuuyl continued @ rtyyw64w
# scan/mixed-[0.01, 0.99]-bart-softmax_continous/2023-12-29_00-57-38
# scan/curriculum-[0.02, 0.9]-bart-softmax/2024-01-17_16-04-47
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.01, 0.99]-bart-softmax/2024-01-18_18-30-39/checkpoints/last.ckpt'"
# DISC='softmax' # 'gumbel' or 'vqvae' or 'softmax'

# 48tov7q4
CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.01, 0.99]-bart-gumbel/2024-01-25_12-04-48/checkpoints/model-0224-0.0965.ckpt'"
DISC='gumbel' # 'gumbel' or 'vqvae' or 'softmax'

# # m5mxh9pm
# CKPT="'/dlabdata1/masani/blocks/logs/training/runs/scan/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-24_10-15-41/checkpoints/model-49184-0.8101.ckpt'"
# DISC='vqvae' # 'gumbel' or 'vqvae' or 'softmax'

BSIZE=256
DEVICE=[0]
NAME="scan_final"
LR=0.001
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=1000


python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.4 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=2 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="unsupsup" logger.wandb.tags=["unsupsup"]

python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
    model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.4 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=2 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="unsupsup" logger.wandb.tags=["unsupsup"]

# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.4 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=2 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="unsupsup" logger.wandb.tags=["unsupsup"]

# python3 run_train.py +experiment=scan_curriculum.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE sequence_to_sequence_model_key=$SEQMODEL discretizer_key=$DISC model.checkpoint_path="$CKPT" \
#     model.substitute_config.model_params.usex=False model.substitute_config.model_params.usez=False \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.4 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=2 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="unsupsup" logger.wandb.tags=["unsupsup"]

