python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
python3 run_train.py +experiment=scan_mixed.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.7] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True || true
