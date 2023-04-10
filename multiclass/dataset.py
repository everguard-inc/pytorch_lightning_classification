import pandas as pd
import numpy as np
import cv2
import os, random
from tqdm import tqdm
import albumentations as A
from config import Config
from typing import NoReturn
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

    
class CustomDataset(Dataset):
    def __init__(self,
                train : bool,
                config : Config,
                transform = None
        ):
        if train:
            images_root = config.train_images_root
            cabble_masks_root = config.train_cabble_masks_root
            remote_masks_root = config.train_remote_masks_root
            labels_root = config.train_labels_root
            df_path = config.train_df_path
        else:
            images_root = config.val_images_root
            cabble_masks_root = config.val_cabble_masks_root
            remote_masks_root = config.val_remote_masks_root
            labels_root = config.val_labels_root
            df_path = config.val_df_path
             
        if config.prepare_dataset:
            CustomDataset.prepare_dataset(
                images_root, 
                cabble_masks_root, 
                remote_masks_root, 
                labels_root,
                df_path
            )
            
        self.df = pd.read_csv(df_path)

        if train:
            self.df = CustomDataset.oversample(self.df, config)

        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def open_txt(names_file_path):
        with open(names_file_path, "r") as text_file:
            rows = text_file.readlines()
        for i, line in enumerate(rows):
            rows[i] = line.replace("\n", "")
        return rows
    
    @staticmethod
    def prepare_dataset(
        images_root, 
        cabble_masks_root, 
        remote_masks_root, 
        labels_root,
        df_path
    ) -> NoReturn:
        df = {
            'image_path': [],
            'cabble_mask_path': [],
            'remote_mask_path': [],
            'x1':[], 'y1':[], 'x2':[], 'y2':[],
            'label':[]
        }
        for label_file_path in tqdm(os.listdir(labels_root)):
            image_path = os.path.join(images_root, label_file_path[:-3] + 'jpg')
            img_size = cv2.imread(image_path).shape[:2]
            cabble_mask_path = os.path.join(cabble_masks_root, label_file_path[:-3] + 'png')
            remote_mask_path = os.path.join(remote_masks_root, label_file_path[:-3] + 'png')
            rows = CustomDataset.open_txt(os.path.join(labels_root,label_file_path))
            for row in rows:
                int_label, xc, yc, w, h = row.split(" ")
                x1 = int((float(xc) - float(w) / 2) * img_size[1])
                y1 = int((float(yc) - float(h) / 2) * img_size[0])
                x2 = int((float(xc) + float(w) / 2) * img_size[1])
                y2 = int((float(yc) + float(h) / 2) * img_size[0])
                df['image_path'].append(image_path)
                df['cabble_mask_path'].append(cabble_mask_path)
                df['remote_mask_path'].append(remote_mask_path)
                df['x1'].append(x1)
                df['y1'].append(y1)
                df['x2'].append(x2)
                df['y2'].append(y2)
                df['label'].append(int(int_label))
        
        df = pd.DataFrame(df)
        df.to_csv(df_path,index=False)
        
    @staticmethod
    def oversample(df : pd.DataFrame,  config : Config) -> pd.DataFrame:
        for label in sorted(df['label'].unique()):
            print(f"{config.label_names[label]} class = ",len(df[df['label'] == label]))
        
        df_class_to_oversample = df[df["label"] == config.label_to_oversample]
        df_class_other = df[df["label"] != config.label_to_oversample]
        
        df_class_oversampled = df_class_to_oversample.sample(int(len(df_class_other)), replace=True)
        df = pd.concat([df_class_oversampled, df_class_other], axis=0)
        df = df.sample(frac=1)
        
        for label in sorted(df['label'].unique()):
            print(f"{config.label_names[label]} oversampled class = ",len(df[df['label'] == label]))
        
        return df
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        cabble_mask_path = row['cabble_mask_path']
        remote_mask_path = row['remote_mask_path']
        label = row['label']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cabble_mask = cv2.imread(cabble_mask_path)
        remote_mask = cv2.imread(remote_mask_path)
        mask = cabble_mask + remote_mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (image.shape[1],image.shape[0])).astype(np.uint8)
        mask[mask > 0] = 1
        combine_image = np.zeros(image.shape,dtype = np.uint8)
        combine_image[row['y1']:row['y2'],row['x1']:row['x2']] = image[row['y1']:row['y2'],row['x1']:row['x2']]
        combine_image += image * mask
        augmented = self.transform(image=combine_image)
        combine_image = augmented['image']
        return {'image':combine_image, 'target': label, 'path': image_path.split('/')[-1][:-4]+'.jpg'}


def get_train_val_data(config):
    train_dataset = CustomDataset(True, config, get_transform(config, 'train'))
    valid_dataset = CustomDataset(False, config, get_transform(config, 'valid'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def get_transform(config, phase: str):
    if phase == 'train':
        return config.train_augs[config.augs_index]
    else:
        return Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=config.img_size['height'], width=config.img_size['width']),
            A.Normalize(),
            ToTensorV2(),
        ])

    




