BSIZE=256
NAME="scan_final"
LR=0.0007
SEQMODEL='bart' # 'gpt2_gpt2' or 'bart'
NUM_EPOCHS=2000
DEVICE=[0]
# define DISC

# python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.6 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=5 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=1500 logger.wandb.notes="mixed" logger.wandb.tags=["mixed"]

# python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.6 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=5 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=1500 logger.wandb.notes="mixed" logger.wandb.tags=["mixed"]

# python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.99] trainer.devices=$DEVICE \
#     datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
#     model.model_params.usex=False model.model_params.usez=False \
#     callbacks.supervision_scheduler.scheduler_xz.hp_init=0.6 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
#     callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
#     callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
#     callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=5 \
#     +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="mixed" logger.wandb.tags=["mixed"]

python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.99] trainer.devices=$DEVICE \
    datamodule.dataset_parameters.batch_size=$BSIZE model/sequence_to_sequence_model=$SEQMODEL model/discretizer=$DISC \
    model.model_params.usex=False model.model_params.usez=False \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=0.6 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=200 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=3 callbacks.supervision_scheduler.scheduler_z.num_training_steps=5 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="mixed" logger.wandb.tags=["mixed"]
