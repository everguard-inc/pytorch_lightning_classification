import torch

class Config:
    images_path = '/home/rodion/dataset_hands_in_gloves_classification/'
    save_dir = 'logs/'
    image_extension = '*.jpg'
    seed = 42
    model_name = 'efficientnet-b4'
    pretrained = True
    img_size = 150
    num_classes = 3
    label_names = ['not_in_gloves','protective_gloves','gloves_unrecognized']
    lr = 0.001
    min_lr = 1e-6
    t_max = 20
    num_epochs = 40
    batch_size = 32
    accum = 1
    precision = 32
    n_fold = 5
    weights_path = 'logs/resnet50/version_0/checkpoints/epoch=22-step=43906.ckpt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
