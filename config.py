import torch

class Config:
    images_path = '/home/rodion/dataset_hands_in_gloves_classification/'
    save_dir = 'logs/'
    image_extension = '*.jpg'
    seed = 42
    model_name = 'resnet50'
    pretrained = True
    img_size = 150
    num_classes = 3
    label_names = ['not_in_gloves','protective_gloves','gloves_unrecognized']
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 40
    batch_size = 32
    accum = 1
    precision = 32
    n_fold = 5
    weights_path = '/home/rodion/pytorch_lightning_classification/logs/resnet50/version_0/checkpoints/epoch=27-step=53451.ckpt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
