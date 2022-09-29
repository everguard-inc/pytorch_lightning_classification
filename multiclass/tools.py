import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os, random
from sklearn.metrics import f1_score
import fnmatch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from config import Config
import torchvision.models as models
from madgrad import MADGRAD
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy

def find_files(directory : str, pattern : str):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.images_path = df['image'].values
        self.labels = df['labels'].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        label = self.labels[idx]
        
        image = cv2.imread(image_path)[:,:,:]
        #image = np.expand_dims(image, axis=-1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': label}
    
    
class CustomDataset1(Dataset):
    def __init__(self, images_root, df, label_names, transform=None):
        self.images_root = images_root
        self.images_path = df['image_path'].values
        self.labels = df['cobbling'].values
        self.transform = transform
        self.label_names = label_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images_path[idx].split('/')[-1][:-4]+'.png'
        image_path = os.path.join(self.images_root,image_path)
        label = self.label_names[self.labels[idx]]

        image = cv2.imread(image_path)[:,:,:]*255
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': label}


def get_train_val_data1(images_path,exstension = '*.jpg'):
    images_list = list(find_files(images_path, exstension))
    random.shuffle(images_list)
    df = pd.DataFrame([])

    for path in images_list:
        for i, label_name in enumerate(Config.label_names):
            if label_name in path.split('_')[-1]:
                label = i
                break
        
        sample = {'image':path,'labels':label}
        df = df.append(sample,ignore_index=True)
        
    print("TRUE class = ",df[df["labels"] == 0].shape)
    print("FALSE class = ",df[df["labels"] == 1].shape)
    
    
    df_class_0 = df[df["labels"] == 0]
    df_class_1 = df[df["labels"] == 1]
    
    df_class_0_under = df_class_0.sample(int(len(df_class_1)), replace=True)
    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    df = df.sample(n = len(df))
    
    print("TRUE class oversampled = ",df[df["labels"] == 0].shape)
    print("FALSE class oversampled = ",df[df["labels"] == 1].shape)
    
    
    sfk = StratifiedKFold(Config.n_fold)
    for train_idx, valid_idx in sfk.split(df['image'], df['labels']):
        df_train = df.iloc[train_idx]
        df_valid = df.iloc[valid_idx]
        break
    
    train_dataset = CustomDataset(df_train, get_transform('train'))
    valid_dataset = CustomDataset(df_valid, get_transform('valid'))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def get_train_val_data2():
    df = pd.read_csv(Config.train_labels_path)
    
    print("cobbling class = ",df[df["cobbling"] == 'cobbling'].shape)
    print("no_cobbling class = ",df[df["cobbling"] == 'no_cobbling'].shape)
    
    
    df_class_0 = df[df["cobbling"] == 'cobbling']
    df_class_1 = df[df["cobbling"] == 'no_cobbling']
    
    df_class_0_under = df_class_0.sample(int(len(df_class_1)), replace=True)
    df = pd.concat([df_class_0_under, df_class_1], axis=0)
    df = df.sample(n = len(df))
    
    print("cobbling class oversampled= ",df[df["cobbling"] == 'cobbling'].shape)
    print("no_cobbling oversampled = ",df[df["cobbling"] == 'no_cobbling'].shape)
    
    val_df = pd.read_csv(Config.val_labels_path)
    
    train_dataset = CustomDataset1(Config.train_images_path, df, Config.label_names, get_transform('train'))
    valid_dataset = CustomDataset1(Config.val_images_path, val_df, Config.label_names, get_transform('valid'))

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def get_transform(phase: str):
    if phase == 'train':
        return Config.train_augs[Config.augs_index]
    else:
        return Compose([
            A.Resize(height=Config.img_size['height'], width=Config.img_size['width']),
            A.Normalize(),#mean = (0.485,), std = (0.229,)),
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
            #self.model._conv_stem.in_channels = 1
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, Config.num_classes)
        else:
            self.model = models.resnet50(pretrained=True)
            #self.model.conv1.in_channels = 1
            self.model.fc = nn.Linear(2048, Config.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class TrainModule(pl.LightningModule):
    def __init__(self, model):
        super(TrainModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
        self.lr = Config.lr
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.nc = Config.num_classes
        self.reset_metrics()

    def reset_metrics(self):
        self.predicts = []
        self.targets = []

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
        if batch_idx==0:
            self.reset_metrics()
        image = batch['image'].to(Config.device)
        target = batch['target'].to(Config.device)
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
        image = batch['image'].to(Config.device)
        target = batch['target'].to(Config.device)
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
        if batch_idx == Config.num_val_batches-1:
            #self.metrics_file = open(os.path.join(Config.save_log_dir,Config.metrics_file), 'a')
            #self.metrics_file.write(f"val epoch {self.current_epoch}: {metrics}\n")
            #self.metrics_file.close()
            if Config.save_best:
                if mean_f1>=self.best_val_f1:
                    torch.save(self.model.state_dict(), f'{Config.save_log_dir}/epoch_{self.current_epoch}_f1_{mean_f1}.pt')
                    if os.path.exists(f'{Config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt'):
                        os.remove(f'{Config.save_log_dir}/epoch_{self.best_epoch}_f1_{self.best_val_f1}.pt')
                    self.best_val_f1 = mean_f1
                    self.best_epoch = self.current_epoch
            else:
                torch.save(self.model.state_dict(), f'{Config.save_log_dir}/epoch_{self.current_epoch}_f1_{mean_f1}.pt')

            for i in range(len(f1_per_class)):
                Config.neptune_run_object['/'.join(['val', list(Config.label_names.keys())[i], 'f1_score'])].log(round(f1_per_class[i],3))

        return loss



