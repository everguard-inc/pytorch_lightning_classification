import torch

class Config:
    images_path = '/home/rodion/labs/da_course/lab4/task2'
    save_dir = 'logs/'
    image_extension = '*.jpg'
    seed = 42
    model_name = 'efficientnet-b3'
    pretrained = True
    img_size = 480
    num_classes = 5
    label_names = ['protective_gloves','rubber_gloves','unusual_gloves','not_in_gloves','gloves_unrecognized']
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 50
    batch_size = 6
    accum = 1
    precision = 32
    n_fold = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')