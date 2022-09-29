import torch
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

class Config:
    train_images_path = '/home/eg/rodion/dataset_cobbling/train/masks/cobbling'
    train_labels_path = '/home/eg/rodion/dataset_cobbling/classification_ann/train.csv'
    val_images_path = '/home/eg/rodion/dataset_cobbling/val/masks/cobbling'
    val_labels_path = '/home/eg/rodion/dataset_cobbling/classification_ann/val.csv'
    save_dir = 'logs/'
    save_log_dir = None
    image_extension = '*.jpg'
    seed = 42
    model_name = 'resnet50'#'efficientnet-b3'
    pretrained = True
    save_best = True
    metrics_file = 'metrics.txt'
    num_classes = 2
    label_names = {'cobbling':0,'no_cobbling':1}
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 100
    batch_size = 9
    img_size = {'height':540, 'width':960}
    accum = 1
    precision = 32
    n_fold = 5
    weights_path = ''
    neptune_run_object = None
    neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YmU0ZjcyOC00ZmQyLTRjY2QtYTE2MS04YzE1NTc0ZDMxNmYifQ=="
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_val_batches = None
    optimizer = "SGD"
    augs_index = 0
    train_augs = [
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Rotate(p=0.5, limit=90),
            A.VerticalFlip(p=0.5),
            #A.HueSaturationValue(always_apply=False, p=0.3, hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-150, 150)),
            A.ElasticTransform(always_apply=False, p=0.3, alpha=4, sigma=100, alpha_affine=10, interpolation=1, border_mode=1),
            #A.MedianBlur(always_apply=False, p=0.2, blur_limit=(11, 21)),
            #A.Downscale(p=0.1),
            A.Normalize(),#mean = (0.485,), std = (0.229,)),
            ToTensorV2(),
        ])
        ]
