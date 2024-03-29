import torch
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

class Config:
    train_images_path = '/home/rodion/crops/dataset_ppe/crops'
    val_images_path = '/home/rodion/crops/dataset_ppe/crops'
    train_df_path = '/home/rodion/crops/dataset_ppe/ann/train.csv'
    val_df_path = '/home/rodion/crops/dataset_ppe/ann/val.csv'
    save_dir = 'logs/'
    save_log_dir = None
    save_best = True
    metrics_file = 'metrics.txt'
    seed = 42
    model_name = 'efficientnet-b3'
    pretrained = True
    img_size = {'height':185, 'width':80}
    label_names = ['in_harness','not_in_harness','harness_unrecognized','in_hardhat','not_in_hardhat',
    'hardhat_unrecognized','in_vest','not_in_vest','vest_unrecognized','person_in_bucket','person_not_in_bucket']
    num_classes = len(label_names)
    lr = 0.0005
    min_lr = 0.000001
    num_kfolds = 1
    current_fold = 0
    t_max = 20
    num_epochs = 20
    batch_size = 30
    accum = 1
    precision = 32
    conf_th = 0.5
    num_train_batches = None
    num_val_batches = None
    optimizer = "Madgrad"
    neptune_run_object = None
    neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1OTFhZDJkYS01NTI1LTQ5MTktODdjYS04OTE0Y2JmNDIzMDYifQ=="
    weights_path = 'weights/5fold_effnet3/'
    ffcv_dataset_path = 'ffcv_converted/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unrec_prob = 0
    augs_index = 0

    unrecognized_augs = Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.RandomSunFlare(flare_roi=(0.3, 0.1, 0.7, 0.3),angle_lower=0,angle_upper=0.2,num_flare_circles_lower=1,num_flare_circles_upper=5, src_radius=200,src_color=(255, 255, 255),always_apply=False,p=1), 
            A.Downscale(p=1)  
        ])

    train_augs = [
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(always_apply=False, p=0.33, hue_shift_limit=(-5, 5), sat_shift_limit=(-5, 5), val_shift_limit=(-150, 150)),
            A.ElasticTransform(always_apply=False, p=0.1, alpha=4, sigma=100, alpha_affine=10, interpolation=1, border_mode=1),
            A.Downscale(p=0.15),  
            A.MedianBlur(always_apply=False, p=0.25, blur_limit=(11, 21)),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.HueSaturationValue(always_apply=False, p=1, hue_shift_limit=(-40, 40), sat_shift_limit=(-50, 50), val_shift_limit=(-70, 70)),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.HueSaturationValue(hue_shift_limit=40,sat_shift_limit=50,val_shift_limit=70,always_apply=False,p=0.5,),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.ElasticTransform(alpha=4,sigma=100,alpha_affine=50,interpolation=1,  border_mode=4,   value=None,
                mask_value=None,always_apply=False,approximate=False,p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),angle_lower=0,angle_upper=1,num_flare_circles_lower=6,num_flare_circles_upper=10,
                                            src_radius=20,src_color=(255, 255, 255),always_apply=False,p=0.5,),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),angle_lower=0,angle_upper=1,num_flare_circles_lower=6,num_flare_circles_upper=10,
                                            src_radius=20,src_color=(255, 255, 255),always_apply=False,p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
        Compose([
            A.Resize(height=img_size['height'], width=img_size['width']),
            A.RandomFog(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]),
    ]
