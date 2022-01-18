import torch

class Config:
    train_images_path = '/home/rodion/crops/train'
    val_images_path = '/home/rodion/crops/val'
    train_df_path = '/home/rodion/crops/train.csv'
    val_df_path = '/home/rodion/crops/val.csv'
    save_dir = 'logs/'
    metrics_file = 'metrics.txt'
    seed = 42
    model_name = 'efficientnet-b3'
    pretrained = True
    img_size = {'height':185, 'width':80}
    label_names = ['in_harness','not_in_harness','harness_unrecognized','in_hardhat','not_in_hardhat',
    'hardhat_unrecognized','in_vest','not_in_vest','person_in_bucket','person_not_in_bucket']
    num_classes = len(label_names)
    lr = 0.0005
    min_lr = 1e-6
    t_max = 20
    num_epochs = 40
    batch_size = 20
    accum = 1
    precision = 32
    n_fold = 5
    conf_th = 0.5
    num_train_batches = None
    num_val_batches = None
    weights_path = 'logs/resnet50/version_0/checkpoints/epoch=22-step=43906.ckpt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
