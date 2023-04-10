import torch
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

class Config:
    train_images_root = '/home/rb/dataset_person_remote/train/images'
    train_cabble_masks_root = '/home/rb/cabble_remote_masks/masks/train/cable'
    train_remote_masks_root = '/home/rb/cabble_remote_masks/masks/train/remote'
    train_labels_root = '/home/rb/dataset_person_remote/train/labels'
    train_df_path = '/home/rb/dataset_person_remote/train_df.csv'
    val_images_root = '/home/rb/dataset_person_remote/val/images'
    val_cabble_masks_root = '/home/rb/cabble_remote_masks/masks/val/cable'
    val_remote_masks_root = '/home/rb/cabble_remote_masks/masks/val/remote'
    val_labels_root = '/home/rb/dataset_person_remote/val/labels'
    val_df_path = '/home/rb/dataset_person_remote/val_df.csv'
    prepare_dataset = False
    save_log_dir = 'logs/'
    image_extension = '*.jpg'
    seed = 42
    model_name = 'resnet34'
    pretrained = True
    save_best = True
    metrics_file = 'metrics.txt'
    num_classes = 3
    label_names = ['holding_remote','not_holding_remote','unrecognized_holding_remote']
    label_to_oversample = 0
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 100
    batch_size = 12
    img_size = {'height':552, 'width':960}
    accum = 1
    precision = 32
    n_fold = 5
    weights_path = '/home/eg/rodion/pytorch_lightning_classification/multiclass/logs/epoch_51_f1_0.951.pt'
    neptune_run_object = None
    neptune_project = "platezhkina13/nucor-remote-cable"
    neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MWU0OTI0ZC00MjlkLTRmYjktYTc5Yi0yOGUzZjVjZGQzZGUifQ=="
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_val_batches = None
    optimizer = "Madgrad"
    augs_index = 0
    train_augs = [
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(always_apply=False, p=0.3, hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-150, 150)),
            A.ElasticTransform(always_apply=False, p=0.3, alpha=4, sigma=100, alpha_affine=10, interpolation=1, border_mode=1),
            A.MedianBlur(always_apply=False, p=0.3, blur_limit=(11, 21)),
            A.Downscale(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
        ]
