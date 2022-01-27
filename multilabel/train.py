from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools import CustomModel, TrainModule, get_datasets, get_transform
from config import Config
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import neptune.new as neptune
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

def get_cur_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")

seed_everything(Config.seed)

def train_cross_val(train_dataset):
    kf = KFold(n_splits=Config.num_kfolds,shuffle=True)
        
    for fold, (train_ids, test_ids) in enumerate(kf.split(train_dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4, sampler=test_subsampler)
        Config.current_fold = fold
        model = CustomModel(model_name=Config.model_name, pretrained=Config.pretrained)
        model = model.to(Config.device)
        logger = CSVLogger(save_dir=Config.save_dir, name=Config.model_name)
        logger.log_hyperparams(Config.__dict__)
        Config.save_log_dir = logger.log_dir
        lit_model = TrainModule(model)                                    
        trainer = Trainer(
            max_epochs=Config.num_epochs,
            gpus=1,
            accumulate_grad_batches=Config.accum,
            precision=Config.precision,
            callbacks=[EarlyStopping(monitor='valid_f1', patience=20, mode='max')],
            logger=logger,
            weights_summary='top',
        )
        Config.num_train_batches = len(train_loader)
        Config.num_val_batches = len(val_loader)

        trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=val_loader)


def train_simple(train_dataset,val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    model = CustomModel(model_name=Config.model_name, pretrained=Config.pretrained)
    model = model.to(Config.device)
    logger = CSVLogger(save_dir=Config.save_dir, name=Config.model_name)
    logger.log_hyperparams(Config.__dict__)
    Config.save_log_dir = logger.log_dir
    lit_model = TrainModule(model)                                    
    trainer = Trainer(
            max_epochs=Config.num_epochs,
            gpus=1,
            accumulate_grad_batches=Config.accum,
            precision=Config.precision,
            callbacks=[EarlyStopping(monitor='valid_f1', patience=20, mode='max')],
            logger=logger,
            weights_summary='top',
        )
    Config.num_train_batches = len(train_loader)
    Config.num_val_batches = len(val_loader)

    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=val_loader)
    

if __name__ == "__main__":
    run = neptune.init(
    project="ever/rlg",
    api_token=Config.neptune_api_token
    )
    run['train_date'] = get_cur_time()
    run['model_name'] = Config.model_name
    run['input_height'] = Config.img_size['height']
    run['input_width'] =  Config.img_size['width']
    run['labels'] = Config.label_names
    run['augmentations_configs'] = str(get_transform('unrecognized_augs'))+str(get_transform('train'))
    
    Config.neptune_run_object = run

    train_dataset, val_dataset = get_datasets()

    if Config.num_kfolds<=1:
        train_simple(train_dataset, val_dataset)
    else:
        train_cross_val(train_dataset)
    

    run.stop()