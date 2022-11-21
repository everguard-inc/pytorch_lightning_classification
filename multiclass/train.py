from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import get_train_val_data, get_transform
from tools import CustomModel, TrainModule
from config import Config
import argparse
import torch
import neptune.new as neptune
from datetime import datetime

seed_everything(Config.seed)

import warnings
warnings.filterwarnings('ignore')

def get_cur_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")

if __name__ == "__main__":
    
    config = Config()

    run = neptune.init(
    project=config.neptune_project,
    api_token=config.neptune_api_token
    )
    run['train_date'] = get_cur_time()
    run['model_name'] = config.model_name
    run['input_height'] = config.img_size['height']
    run['input_width'] =  config.img_size['width']
    run['labels'] = config.label_names
    run['augmentations_configs'] = str(get_transform(config, 'train'))
    
    config.neptune_run_object = run

    model = CustomModel(model_name=config.model_name, config = config, pretrained=config.pretrained)
    model = model.to(config.device)
    logger = CSVLogger(save_dir=Config.save_log_dir, name=Config.model_name)
    logger.log_hyperparams(Config.__dict__)

    lit_model = TrainModule(model, config)

    trainer = Trainer(
        max_epochs=Config.num_epochs,
        gpus=1,
        accumulate_grad_batches=Config.accum,
        precision=Config.precision,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=20, mode='min')],
    )

    train_loader, valid_loader = get_train_val_data(config)
    Config.num_val_batches = len(valid_loader)

    trainer.fit(lit_model, train_loader, valid_loader)

    run.stop()