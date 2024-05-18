BSIZE=200
NAME="scan_final"
LR=0.0005
NUM_EPOCHS=2000

python3 run_train.py +experiment=scan_blocks_library.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.99]\
    datamodule.dataset_parameters.batch_size=$BSIZE trainer.devices=[0]\
    model.model_params.usex=False model.model_params.usez=False \
    callbacks.supervision_scheduler.scheduler_xz.hp_init=1.0 callbacks.supervision_scheduler.scheduler_xz.hp_end=0.6 \
    callbacks.supervision_scheduler.scheduler_z.hp_init=1.0 callbacks.supervision_scheduler.scheduler_z.hp_end=1.0 \
    callbacks.supervision_scheduler.scheduler_xz.num_warmup_steps=10 callbacks.supervision_scheduler.scheduler_xz.num_training_steps=100 \
    callbacks.supervision_scheduler.scheduler_z.num_warmup_steps=1 callbacks.supervision_scheduler.scheduler_z.num_training_steps=100 \
    +test=True model.optimizer.lr=$LR name=$NAME model.lr_scheduler.monitor="val/loss" num_epochs=$NUM_EPOCHS logger.wandb.notes="supunsup" logger.wandb.tags=["supunsup"]

 