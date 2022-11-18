import os
import cv2
import timm
import torch
import torch.nn as nn
from config import Config
from copy import deepcopy
from madgrad import MADGRAD
import pytorch_lightning as pl
import numpy as np
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class CustomModel(nn.Module):
    def __init__(self, model_name='resnet50', config = Config(), pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained = pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features,config.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class TrainModule(pl.LightningModule):
    def __init__(self, model, config):
        super(TrainModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.lr = config.lr
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.nc = config.num_classes
        self.reset_metrics()

    def reset_metrics(self):
        self.predicts = []
        self.targets = []

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.config.optimizer == "Madgrad":
            self.optimizer = MADGRAD(self.model.parameters(),lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.t_max, eta_min=Config.min_lr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        if batch_idx==0:
            self.reset_metrics()
        image = batch['image'].to(self.config.device)
        target = batch['target'].to(self.config.device)
        target = target.long()
        output = self.model(image)
        loss = self.criterion(output, target)
        pt = torch.exp(-loss)
        loss = (0.25 * (1-pt)**2 * loss).mean()
        output = torch.sigmoid(output)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()
        for pred in predicted:
            self.predicts.append(pred)
        for t in target:
            self.targets.append(t)
        metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
        metrics_per_class = []
        f1_per_class = []
        for _ in range(self.nc):
            metrics_per_class.append(deepcopy(metrics_dict))
        for i, label in enumerate(self.predicts):
            if label == self.targets[i]:
                metrics_per_class[label]['tp']+=1
            else:
                metrics_per_class[label]['fp']+=1
                metrics_per_class[self.targets[i]]['fn']+=1
        for i in range(self.nc):
            pr_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fp'] + 1e-9)
            recall_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fn'] + 1e-9)
            f1_per_class.append(round(2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9),3))
        
        logs = {'train_loss': loss, 'train_f1': round(np.array(f1_per_class).mean(),3), 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx==0:
            self.reset_metrics()
        image = batch['image'].to(self.config.device)
        target = batch['target'].to(self.config.device)
        target = target.long()
        output = self.model(image)
        loss = self.criterion(output, target)
        pt = torch.exp(-loss)
        loss = (0.25 * (1-pt)**2 * loss).mean()
        output = torch.sigmoid(output)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()
        for pred in predicted:
            self.predicts.append(pred)
        for t in target:
            self.targets.append(t)
        metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
        metrics_per_class = []
        f1_per_class = []
        for _ in range(self.nc):
            metrics_per_class.append(deepcopy(metrics_dict))
        for i, label in enumerate(self.predicts):
            if label == self.targets[i]:
                metrics_per_class[label]['tp']+=1
            else:
                metrics_per_class[label]['fp']+=1
                metrics_per_class[self.targets[i]]['fn']+=1
        for i in range(self.nc):
            pr_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fp'] + 1e-9)
            recall_05 = metrics_per_class[i]['tp'] / (metrics_per_class[i]['tp'] + metrics_per_class[i]['fn'] + 1e-9)
            f1_per_class.append(round(2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9),3))
        
        mean_f1 = round(np.array(f1_per_class).mean(),3)
        logs = {'valid_loss': loss, 'valid_f1': mean_f1}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == self.config.num_val_batches-1:
            #self.metrics_file = open(os.path.join(Config.save_log_dir,Config.metrics_file), 'a')
            #self.metrics_file.write(f"val epoch {self.current_epoch}: {metrics}\n")
            #self.metrics_file.close()
            if self.config.save_best:
                if mean_f1>=self.best_val_f1:
                    torch.save(self.model.state_dict(), f'{self.config.save_log_dir}/epoch_{self.current_epoch}_f1_{mean_f1}.pt')
                    if os.path.exists(f'{self.config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt'):
                        os.remove(f'{self.config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt')
                    self.best_val_f1 = mean_f1
                    self.best_epoch = self.current_epoch
            else:
                torch.save(self.model.state_dict(), f'{self.config.save_log_dir}/epoch_{self.current_epoch}_f1_{mean_f1}.pt')

            for i in range(len(f1_per_class)):
                self.config.neptune_run_object['/'.join(['val', self.config.label_names[i], 'f1_score'])].log(round(f1_per_class[i],3))

        return loss