BSIZE=512
DEVICE=[0]

python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.99,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.01,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.02,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.04,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
python3 run_train.py +experiment=sfst_suponly.yaml datamodule.dataset_parameters.supervision_ratio=[0.08,0.9] model/discretizer=$DISC trainer.devices=$DEVICE datamodule.dataset_parameters.batch_size=$BSIZE +test=True name=$NAME || true
