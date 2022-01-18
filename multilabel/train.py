from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools import CustomModel, TrainModule, get_train_val_data, get_transform
from config import Config
import neptune.new as neptune
from datetime import datetime

def get_cur_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")

seed_everything(Config.seed)

if __name__ == "__main__":

    run = neptune.init(
    project="vova.xnz/MultilabelClassifierRLG",
    api_token=Config.neptune_api_token
    )

    run['train_date'] = get_cur_time()
    run['model_name'] = Config.model_name
    run['input_height'] = Config.img_size['height']
    run['input_width'] =  Config.img_size['width']
    run['labels'] = Config.label_names
    run['augmentations_configs'] = str(get_transform('train'))

    Config.neptune_run_object = run

    model = CustomModel(model_name=Config.model_name, pretrained=Config.pretrained)
    model = model.to(Config.device)
    logger = CSVLogger(save_dir=Config.save_dir, name=Config.model_name)
    logger.log_hyperparams(Config.__dict__)
    Config.save_log_dir = logger.log_dir
    lit_model = TrainModule(model)
    checkpoint_callback = ModelCheckpoint(monitor='valid_f1',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=True,
                                        filename='best',
                                        verbose=False,
                                        mode='max')
                                        

    

    trainer = Trainer(
        max_epochs=Config.num_epochs,
        gpus=1,
        accumulate_grad_batches=Config.accum,
        precision=Config.precision,
        callbacks=[EarlyStopping(monitor='valid_f1', patience=10, mode='max')],
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        weights_summary='top',
    )
    train_loader, valid_loader = get_train_val_data()

    Config.num_train_batches = len(train_loader)
    Config.num_val_batches = len(valid_loader)

    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=valid_loader)

    run.stop()