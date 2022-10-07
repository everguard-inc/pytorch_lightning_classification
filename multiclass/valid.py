from tools import CustomModel, TrainModule, get_train_val_data2
from config import Config
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

if __name__ == "__main__":
    
    model = CustomModel(model_name=Config.model_name, pretrained=False)
    state_dict = torch.load(Config.weights_path)['state_dict']
    state_dict2 = OrderedDict()
    for item in state_dict:
        state_dict2.update({item[6:]:state_dict[item]})
    model.load_state_dict(state_dict2)
    model = model.to(Config.device)
    model.eval()
    train_loader, valid_loader = get_train_val_data2(Config.images_path,Config.image_extension)
    predicts, targets = [], []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            data = batch['image'].to(Config.device)
            target = batch['target']
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            for pred in predicted:
                predicts.append(pred)
            for t in target:
                targets.append(t)

    metrics = list(f1_score(targets, predicts, average=None))
    metrics_file = open('metrics.txt', 'w')
    for i, f1 in enumerate(metrics):
        print(Config.label_names[i], ' = ',f1.round(2))
        metrics_file.write(Config.label_names[i] + ' = ' + str(f1.round(2)) + '\n')
    metrics_file.close()
