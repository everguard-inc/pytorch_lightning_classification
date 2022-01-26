from operator import index
from config import Config
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
import fnmatch
from copy import deepcopy
import pandas as pd
from collections import OrderedDict
from tools import CustomModel, get_train_test_datasets, \
    get_test_dataloader, get_metrics, get_average_score

def find_files(directory : str, pattern : str):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def blend_weights(best_weights_path):
    state_dict = torch.load(best_weights_path[0])
    new_state_dict = deepcopy(state_dict)
    for path in best_weights_path[1:]:
        state_dict = torch.load(path)
        for key in new_state_dict:
            new_state_dict[key] += state_dict[key]
    for key in new_state_dict:
        new_state_dict[key]=(new_state_dict[key]/len(best_weights_path)).float()
    
    return new_state_dict

def parse_best_weights():
    folds_dirs = os.listdir(Config.weights_path)
    all_weights_path = []
    for dir in folds_dirs:
        fold_weights = list(find_files(os.path.join(Config.weights_path,dir), '*.pt'))
        all_weights_path.append(fold_weights)
    all_epoch_score_tuples = []
    for fold_weights in all_weights_path:
        split_epoch_f1score = lambda x: (int(x.split('/')[-1].split('_')[1]),float('.'.join(x.split('/')[-1].split('_')[-1].split('.')[:-1])))
        epoch_score_tuples = [split_epoch_f1score(x) for x in fold_weights]
        all_epoch_score_tuples.append(epoch_score_tuples)
    best_weights_path = []
    for fold_i,fold_weights in enumerate(all_epoch_score_tuples):
        fold_weights.sort(key=lambda x:x[1])
        for path in all_weights_path[fold_i]:
            if 'epoch_'+str(fold_weights[-1][0]) in path:
                best_weights_path.append(path)

    return best_weights_path


def test_metrics(test_loader):

    model = CustomModel(model_name=Config.model_name, pretrained=False)
    best_weights_path = parse_best_weights()
    blend_state_dict = blend_weights(best_weights_path)
    torch.save(blend_state_dict, f'{Config.weights_path}/best.pt')
    model.load_state_dict(blend_state_dict)
    model = model.to(Config.device)
    model.eval()
    metrics = []
    metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
    test_metrics_df = pd.DataFrame()
    for _ in range(Config.num_classes):
        metrics.append(deepcopy(metrics_dict))
        f1_metrics = [0 for i in range(Config.num_classes)]

    with torch.no_grad():
        for batch in tqdm(test_loader):
            data = batch[0].to(Config.device)
            target = batch[1]
            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            metrics = get_metrics(outputs,target,metrics,Config.conf_th)
        
    _,f1_metrics = get_average_score(metrics,f1_metrics)
    for i in range(len(f1_metrics)):
        print(Config.label_names[i], ' = ', f1_metrics[i])
        test_metrics_df[Config.label_names[i]] = pd.Series(f1_metrics[i])
    test_metrics_df.to_csv(f'{Config.weights_path}/test_metrics.csv',index = False)


if __name__ == "__main__":
    train_dataset, test_dataset = get_train_test_datasets()
    test_loader = get_test_dataloader(test_dataset)
    test_metrics(test_loader)