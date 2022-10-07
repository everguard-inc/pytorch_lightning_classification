from tools import CustomModel, TrainModule, get_train_val_data2
from config import Config
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import cv2
import os

if __name__ == "__main__":
    
    model = CustomModel(model_name=Config.model_name, pretrained=False)
    state_dict = torch.load(Config.weights_path)
    state_dict2 = OrderedDict()
    for item in state_dict:
        state_dict2.update({item[:]:state_dict[item]})
    model.load_state_dict(state_dict2)
    model = model.to(Config.device)
    model.eval()
    train_loader, valid_loader = get_train_val_data2()
    label_names = {v: k for k, v in Config.label_names.items()}
    predicts, targets = [], []
    with torch.no_grad():
        for b_i,batch in enumerate(valid_loader):
            data = batch['image'].to(Config.device)
            target = batch['target']
            paths = batch['path']
            output = model(data)
            output = torch.sigmoid(output)
            prob, predicted = torch.max(output.data, 1)
            predicted = predicted.detach().cpu().numpy()
            for pred in predicted:
                predicts.append(pred)
            for t in target:
                targets.append(t)
            for i in range(len(predicted)):
                pred = predicted[i]
                path = paths[i]
                image_path = os.path.join('/home/eg/rodion/dataset_cobbling/val/img',path)
                print(image_path)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"viz/{b_i}_{i}__{label_names[pred]}.jpg",image)
    print(targets, predicts)
    metrics = list(f1_score(targets, predicts, average=None))
    print(metrics)