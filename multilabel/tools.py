import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from config import Config
import torchvision.models as models
from copy import deepcopy



class CustomDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.images_path = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def encode_labels(labels,num_classes):
        encoded = np.zeros(num_classes)
        for l in labels:
            encoded[l] = 1

        return encoded

    @staticmethod
    def decode_labels(labels):
        labels_cat = []
        for i,label in enumerate(labels):
            if label==1:
                labels_cat.append(i)
        return labels_cat

    @staticmethod
    def labels_string_to_int(labels):
        labels_ints = []
        for label in labels:
            for i, label_name in enumerate(Config.label_names):
                if label==label_name:
                    labels_ints.append(i)
                    break
        return labels_ints

    def __getitem__(self, idx):
        labels = self.df[self.df['image_path']==os.path.join(self.path.split('/')[-1],self.images_path[idx])][['harness','hardhat','vest','person_in_bucket']].values.squeeze()
        labels = CustomDataset.labels_string_to_int(labels)
        labels = CustomDataset.encode_labels(labels,Config.num_classes)
        image = cv2.imread(os.path.join(self.path,self.images_path[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': labels}

def get_train_val_data():
    train_df = pd.read_csv(Config.train_df_path)
    val_df = pd.read_csv(Config.val_df_path)

    train_dataset = CustomDataset(train_df, Config.train_images_path, get_transform('train'))
    valid_dataset = CustomDataset(val_df, Config.val_images_path ,get_transform('valid'))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def get_transform(phase: str):
    if phase == 'train':
        return Compose([
            A.Resize(height=Config.img_size['height'], width=Config.img_size['width']),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.GaussNoise(p = 0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomFog(p=0.3),
            A.Rotate(p=1, limit=90),
            A.VerticalFlip(p=0.5),
            A.RandomContrast(limit=0.5, p = 1),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return Compose([
            A.Resize(height=Config.img_size['height'], width=Config.img_size['width']),
            A.Normalize(),
            ToTensorV2(),
        ])
    
class CustomModel(nn.Module):
    def __init__(self, model_name='efficientnet-b3', pretrained=True):
        super().__init__()
        if 'efficientnet' in model_name:
            if pretrained:
                self.model = EfficientNet.from_pretrained(model_name)
            else:
                self.model = EfficientNet.from_name(model_name)
            for param in self.model.parameters():
                param.requires_grad = True
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, Config.num_classes)
        else:
            self.model = models.resnet50(pretrained=True)
            self.model.fc.out_features = Config.num_classes

    def forward(self, x):
        x = self.model(x)
        return x

class TrainModule(pl.LightningModule):
    def __init__(self, model):
        super(TrainModule, self).__init__()
        self.model = model
        self.criterion = nn.BCELoss()
        self.lr = Config.lr
        self.metrics_file = open(Config.metrics_file, 'a')
        self.reset_metrics()
        
    def reset_metrics(self):
        self.metrics = []
        self.metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
        for _ in range(Config.num_classes):
            self.metrics.append(deepcopy(self.metrics_dict))
            self.f1_metrics = [0 for i in range(Config.num_classes)]

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=Config.t_max, eta_min=Config.min_lr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        image = batch['image'].to(Config.device)
        target = batch['target'].to(Config.device)
        target = target.float()
        output = self.model(image)
        output = torch.sigmoid(output)
        loss = self.criterion(output, target)
        if batch_idx==0:
            self.reset_metrics()
        self.metrics = self.get_metrics(output,target,self.metrics,Config.conf_th)
        score = self.get_average_score(self.metrics,self.f1_metrics)
        logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == Config.num_train_batches-1:
            self.metrics_file.write(f"train epoch {self.current_epoch}: {self.f1_metrics}\n")
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'].to(Config.device)
        target = batch['target'].to(Config.device)
        target = target.float()
        output = self.model(image)
        output = self.model(image)
        output = torch.sigmoid(output)
        loss = self.criterion(output, target)
        if batch_idx==0:
            self.reset_metrics()
        self.metrics = self.get_metrics(output,target,self.metrics,Config.conf_th)
        score = self.get_average_score(self.metrics,self.f1_metrics)
        logs = {'valid_loss': loss, 'valid_f1': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == Config.num_val_batches-1:
            self.metrics_file.write(f"val epoch {self.current_epoch}: {self.f1_metrics}\n")

        return loss


    def get_metrics(self, outputs, targets, metrics, conf_th):
        for i, predicted_labels in enumerate(outputs):
            target_labels = targets[i]
            target_labels = (target_labels > conf_th).nonzero().squeeze()
            predicted_labels = (predicted_labels > conf_th).nonzero().squeeze()
            for label in predicted_labels:
                if label in target_labels:
                    metrics[label]['tp']+=1
                else:
                    metrics[label]['fp']+=1
                    metrics[label]['fn']+=1

        return metrics
                
    def get_average_score(self, metrics, f1_metrics):
        for i in range(Config.num_classes):
            pr_05 = metrics[i]['tp'] / (metrics[i]['tp'] + metrics[i]['fp'] + 1e-9)
            recall_05 = metrics[i]['tp'] / (metrics[i]['tp'] + metrics[i]['fn'] + 1e-9)
            f1_metrics[i] = 2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9)   

        return np.mean(f1_metrics)
