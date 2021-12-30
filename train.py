from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools import CustomModel, TrainModule, get_train_val_data
from config import Config
import argparse

seed_everything(Config.seed)


if __name__ == "__main__":

    model = CustomModel(model_name=Config.model_name, pretrained=Config.pretrained)
    lit_model = TrainModule(model)
    logger = CSVLogger(save_dir=Config.save_dir, name=Config.model_name)
    logger.log_hyperparams(Config.__dict__)
    checkpoint_callback = ModelCheckpoint(monitor='valid_f1',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=True,
                                        filename='checkpoint/{epoch:02d}-{valid_f1:.4f}',
                                        verbose=False,
                                        mode='max')

    trainer = Trainer(
        max_epochs=Config.num_epochs,
        gpus=1,
        accumulate_grad_batches=Config.accum,
        precision=Config.precision,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=3, mode='min')],
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        weights_summary='top',
    )

    train_loader, valid_loader = get_train_val_data(Config.images_path,'*.jpeg')

    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=valid_loader)