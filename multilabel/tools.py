from asyncio.log import logger
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import os
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from config import Config
import torchvision.models as models
from madgrad import MADGRAD
import random
from copy import deepcopy


def generate_percent(prob = Config.unrec_prob):
    if random.randint(0,100) < prob:
        return True
    else:
        return False


class CustomDataset(Dataset):
    def __init__(self, images, labels, path, transform = None, unrecognized_transform = None, train = True):
        self.images = images
        self.labels = labels
        self.path = path
        self.transform = transform
        self.unrecognized_transform = unrecognized_transform
        self.train = train

    def __len__(self):
        return len(self.images)

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

    @staticmethod
    def change_labels(labels):
        new_labels = []
        for l in labels:
            if l=='in_hardhat' or l=='not_in_hardhat':
                new_labels.append('hardhat_unrecognized')
            else:
                new_labels.append(l)
            
        return new_labels

    def __getitem__(self, idx):
        labels = self.labels[idx]
        image = cv2.imread(os.path.join(self.path,self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            if generate_percent():
                labels = CustomDataset.change_labels(labels)
                augmented = self.unrecognized_transform(image=image)
                image = augmented['image']
        labels = CustomDataset.labels_string_to_int(labels)
        labels = CustomDataset.encode_labels(labels,Config.num_classes)
        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': labels}


def get_datasets():
    train_df = pd.read_csv(Config.train_df_path)
    val_df = pd.read_csv(Config.val_df_path)

    train_images = train_df['image_path'].values
    train_labels = train_df[['harness','hardhat','vest','person_in_bucket']].values
    val_images = val_df['image_path'].values
    val_labels = val_df[['harness','hardhat','vest','person_in_bucket']].values

    train_dataset = CustomDataset(train_images, train_labels, Config.train_images_path, get_transform('train'),get_unrecognized_transform(),True)
    test_dataset = CustomDataset(val_images, val_labels, Config.val_images_path ,get_transform('valid'),None, False)

    return train_dataset, test_dataset

def get_unrecognized_transform():
    return Config.unrecognized_augs

def get_transform(phase: str):
    if phase == 'train':
        return Config.train_augs[Config.augs_index]
    else:
        return Compose([
            A.Resize(height=Config.img_size['height'], width=Config.img_size['width']),
            A.Normalize(),
            ToTensorV2(),
        ])

    
class CustomModel(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        if 'efficientnet' in model_name:
            if pretrained:
                self.model = EfficientNet.from_pretrained(model_name)
            else:
                self.model = EfficientNet.from_name(model_name)
            for param in self.model.parameters():
                param.requires_grad = True
            self.model = EfficientNet.from_name('efficientnet-b3')
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, Config.num_classes)
        else:
            self.model = models.resnet50(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, Config.num_classes)
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

class TrainModule(pl.LightningModule):
    def __init__(self, model):
        super(TrainModule, self).__init__()
        self.model = model
        self.criterion = nn.BCELoss()
        self.lr = Config.lr
        self.best_val_f1 = 0
        self.best_epoch = 0
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
        if Config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif Config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif Config.optimizer == "Madgrad":
            self.optimizer = MADGRAD(self.model.parameters(),lr=self.lr)

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
        score,self.f1_metrics = self.get_average_score(self.metrics,self.f1_metrics)
        logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == Config.num_train_batches-1:
            self.metrics_file = open(os.path.join(Config.save_log_dir,Config.metrics_file), 'a')
            self.metrics_file.write(f"train epoch {self.current_epoch}: {self.f1_metrics}\n")
            self.metrics_file.close()
            for i in range(len(self.f1_metrics)):
                Config.neptune_run_object['/'.join(['metrics', 'train',Config.label_names[i], 'f1_score'])].log(self.f1_metrics[i])
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'].to(Config.device)
        target = batch['target'].to(Config.device)
        target = target.float()
        output = self.model(image)
        output = torch.sigmoid(output)
        loss = self.criterion(output, target)
        if batch_idx==0:
            self.reset_metrics()
        self.metrics = self.get_metrics(output,target,self.metrics,Config.conf_th)
        score,self.f1_metrics = self.get_average_score(self.metrics,self.f1_metrics)
        logs = {'valid_loss': loss, 'valid_f1': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == Config.num_val_batches-1:
            self.metrics_file = open(os.path.join(Config.save_log_dir,Config.metrics_file), 'a')
            self.metrics_file.write(f"val epoch {self.current_epoch}: {self.f1_metrics}\n")
            self.metrics_file.close()
            if Config.save_best:
                if score>=self.best_val_f1:
                    torch.save(self.model.state_dict(), f'{Config.save_log_dir}/epoch_{self.current_epoch}_f1_{score}.pt')
                    if os.path.exists(f'{Config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt'):
                        os.remove(f'{Config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt')
                    self.best_val_f1 = score
                    self.best_epoch = self.current_epoch
            else:
                torch.save(self.model.state_dict(), f'{Config.save_log_dir}/epoch_{self.current_epoch}_f1_{score}.pt')

            for i in range(len(self.f1_metrics)):
                Config.neptune_run_object['/'.join(['metrics', 'val', Config.label_names[i], 'f1_score'])].log(self.f1_metrics[i])

        return loss

    def get_metrics(self, outputs, targets, metrics, conf_th):
        for i, predicted_labels in enumerate(outputs):
            target_labels = targets[i]
            target_labels = (target_labels > conf_th).nonzero().squeeze()
            predicted_labels = (predicted_labels > conf_th).nonzero().squeeze()
            try:
                for label in predicted_labels:
                    if label in target_labels:
                        metrics[label]['tp']+=1
                    else:
                        metrics[label]['fp']+=1
                        metrics[label]['fn']+=1
            except:
                ...

        return metrics

    def get_average_score(self, metrics, f1_metrics):
        for i in range(Config.num_classes):
            pr_05 = metrics[i]['tp'] / (metrics[i]['tp'] + metrics[i]['fp'] + 1e-9)
            recall_05 = metrics[i]['tp'] / (metrics[i]['tp'] + metrics[i]['fn'] + 1e-9)
            f1_metrics[i] = round(2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9),3)   

        f1_mean_labels = [0,1,3,4,6,7]
        mean = 0
        for l in f1_mean_labels:
            mean+=f1_metrics[l]

        return round(mean/len(f1_mean_labels),3),f1_metrics