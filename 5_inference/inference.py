import os
import random
import warnings
import pandas as pd
import monai
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from monai.transforms import Activations
from torch import nn, optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score


from 2_dataload.3_Dataset import DTDataset
from 3_model.MAM_model import DualTowerResNet10
warnings.filterwarnings("ignore", message=".*AddChannel.*", category=FutureWarning)




def main():
    """Device Configuration"""
    device_count = torch.cuda.device_count()
    if device_count > 0:
        print(f"Available GPUs: {device_count}")
        if device_count == 1:
            device = torch.device("cuda:0")
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cuda:0")
            print("Using all available GPUs:")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        print("No GPUs found, using CPU.")

    folds = [0, 1, 2, 3, 4]
    epoch_num = 'Best_Model_After100'
    for fold in folds:
        val_path = '1_dataset/test.csv'
        weight_path = f"save_model\\{fold}\\{epoch_num}.pth"
        save_path = f"\\MAM_prediction_fold{fold}.csv"

        print(f'fold：{fold}')

        # Transform Setting
        data_transform = monai.transforms.Compose()
        val_dataset = DTDataset(val_path, transform=data_transform)
        val_num = len(val_dataset)
        val_p_count = 0
        for samples2 in val_dataset:
            val_p_count += samples2['label']
        val_p_ratio = round(val_p_count / val_num * 100, 2)
        print("using {} images including {}({}%) positive samples for validating".format(val_num, val_p_count, val_p_ratio))

        # Dataloder Configuration
        batch_size = 1
        nw = 1
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, drop_last=False,
                                pin_memory=torch.cuda.is_available())
        """Model Configuration"""
        # Load Weights
        net = DualTowerResNet10(num_class=1)
        weights = torch.load(weight_path)
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        net.load_state_dict(weights)
        net = nn.DataParallel(net.to(device), device_ids=[0])

        net.eval()
        with torch.no_grad():
            val_y_pred = torch.tensor([], dtype=torch.float32, device=device)
            val_y = torch.tensor([], dtype=torch.float32, device=device)
            for val_data in val_loader:
                val_imgA, val_imgB, val_labels = \
                        val_data['imgA'].float().to(device), \
                        val_data['imgB'].float().to(device), \
                        val_data['label'].float().to(device)
                val_output = net(val_imgA, val_imgB).squeeze()
                val_output_prob = torch.sigmoid(val_output).squeeze().unsqueeze(0) # Apply Sigmoid function
                val_y_pred = torch.cat([val_y_pred, val_output_prob], dim=0)
                val_y = torch.cat([val_y, val_labels], dim=0)

            val_y_pred_np = val_y_pred.cpu().numpy()
            val_y_np = val_y.cpu().numpy()
            val_auc_result = roc_auc_score(val_y_np, val_y_pred_np)
            print(val_auc_result)
            results_df = pd.DataFrame({
                'label': val_y_np,
                'Predicted': val_y_pred_np,
            })
            results_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    main()
