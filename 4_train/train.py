import os
import random
import warnings
import json
import monai
import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import RandRotate, RandFlip, RandGaussianNoise, Activations, AsDiscrete, AddChannel, \
    RandAdjustContrast, RandCropByPosNegLabel, RandShiftIntensityd, RandAffine, Rand3DElastic
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
from 2_dataload.2_Dataset import DTDataset
from sklearn.metrics import accuracy_score, roc_auc_score
from 3_model.MAM_model import DualTowerResNet10




logs = []
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
def custom_print(*args, **kwargs):
    message = " ".join(map(str, args))
    logs.append(message)
    print(*args, **kwargs)
def save_logs_to_json(filename):
    with open(filename, 'w') as f:
        json.dump(logs, f, indent=4)
def main():
    """Set_seed"""
    seed = 1
    set_seed(seed)
    """Device Configuration"""
    device_count = torch.cuda.device_count()
    if device_count > 0:
        custom_print(f"Available GPUs: {device_count}")
        if device_count == 1:
            device = torch.device("cuda:0")
            custom_print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cuda:0")
            custom_print("Using all available GPUs:")
            for i in range(device_count):
                custom_print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device("cpu")
        custom_print("No GPUs found, using CPU.")

    """Data Configuration"""
    log_id = 1001
    logdir = f"logs/{log_id}"
    save_path = f'save_model/{log_id}'
    logjson_savepath = f"json_log/{log_id}_log.json"

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    # Data Load & Split
    current_fold = log_id % 10
    data_root_path = '1_dataset/'
    train_path = os.path.join(data_root_path, f'train_fold{current_fold}.csv')
    val_path = os.path.join(data_root_path, f'val_fold{current_fold}.csv')

    # Transform Setting
    probe = 0.3
    data_transform = {
        "train": monai.transforms.Compose([RandRotate(prob=probe, range_x=0.3, range_y=0.3, range_z=0.3),
                                           RandGaussianNoise(prob=0.3, mean=0.0, std=0.2),
                                           RandAdjustContrast(prob=probe, gamma=(0.2, 2)),
                                           Rand3DElastic(prob=probe, sigma_range=(5, 8), magnitude_range=(100, 200))
                                           ]),
        "val": monai.transforms.Compose()}
    # Dataset Reading
    train_dataset = DTDataset(train_path, transform=data_transform["train"])
    train_num = len(train_dataset)
    train_p_count = 0
    for samples1 in train_dataset:
        train_p_count += samples1['label']
    train_p_ratio = round(train_p_count / train_num * 100, 2)
    val_dataset = DTDataset(val_path, transform=data_transform["val"])
    val_num = len(val_dataset)
    val_p_count = 0
    for samples2 in val_dataset:
        val_p_count += samples2['label']
    val_p_ratio = round(val_p_count / val_num * 100, 2)
    custom_print("using {} images including {}({}%) positive samples for training".format(train_num, train_p_count,
                                                                                          train_p_ratio))
    custom_print(
        "using {} images including {}({}%) positive samples for validating".format(val_num, val_p_count, val_p_ratio))

    # Dataloder Configuration
    batch_size = 16
    num_samples = 500
    nw = 2
    custom_print('Using {} dataloader workers every process'.format(nw))
    weights = [4 if sample['label'] == 1 else 1 for sample in train_dataset]
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=nw, drop_last=True,
                              sampler=sampler, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, drop_last=False,
                            pin_memory=torch.cuda.is_available())


    """Model Configuration"""
    # Load Weights
    net = DualTowerResNet10(num_class=1)
    path1 = "model/resnet_10.pth"
    path2 = "model/resnet_10.pth"
    net.load_pretrained_weights(path1, path2)
    net = nn.DataParallel(net.to(device), device_ids=[0, 1])


    """Training Setting"""
    learning_rate = 1e-5
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    auto_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=15)
    epochs = 150
    val_interval = 1
    best_metric = -1
    best_metric_100 = -1
    best_metric_epoch_100 = -1
    best_auc = -1
    epochs_no_improve = 0
    best_metric_epoch = -1
    tresh_hold = 0.5
    step_lr_list = []
    writer = SummaryWriter(log_dir=logdir)

    """Training Model"""
    for epoch in range(epochs):
        """Training"""
        custom_print('-' * 10)
        custom_print(f"epoch {epoch + 1}/{epochs}")
        net.train()
        train_label_num = []
        epoch_loss = 0.0
        step = 0
        for batch_data in train_loader:
            step += 1
            imgA, imgB, labels = batch_data['imgA'].float().to(device), \
                batch_data['imgB'].float().to(device), \
                batch_data['label'].float().to(device)
            train_label_num += labels.tolist()
            optimizer.zero_grad()
            outputs = net(imgA, imgB).squeeze()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        train_label_count = sum(train_label_num)
        train_label_percentage = (train_label_count / len(train_label_num)) * 100
        custom_print(
            'epoch {}: positive label_count: {}({:.2f}%)'.format(epoch + 1, train_label_count, train_label_percentage))
        epoch_loss /= step
        writer.add_scalars('Loss', {"average_train_loss": epoch_loss}, epoch + 1)
        writer.add_scalars('Learning Rate', {'current_learning_rate': optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch + 1)
        custom_print(f"epoch {epoch + 1} average_train loss: {epoch_loss:.4f}")
        custom_print('----------current_lr-----------', optimizer.state_dict()['param_groups'][0]['lr'])

        # Validate
        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                # Train set evaluation
                train_y_pred = torch.tensor([], dtype=torch.float32, device=device)
                train_y = torch.tensor([], dtype=torch.float32, device=device)
                for train_data in train_loader:
                    train_imgA, train_imgB, train_labels = train_data['imgA'].float().to(device), \
                        train_data['imgB'].float().to(device), \
                        train_data['label'].float().to(device)
                    train_output = net(train_imgA, train_imgB).squeeze()
                    train_output_prob = torch.sigmoid(train_output)
                    train_y_pred = torch.cat([train_y_pred, train_output_prob], dim=0)
                    train_y = torch.cat([train_y, train_labels], dim=0)

                train_y_pred_np = train_y_pred.cpu().numpy()
                train_y_np = train_y.cpu().numpy()
                train_acc_metric = accuracy_score(train_y_np, train_y_pred_np > tresh_hold)
                train_auc_result = roc_auc_score(train_y_np, train_y_pred_np)

                custom_print(
                    "current epoch: {} train accuracy: {:.4f} train AUC: {:.4f}".format(
                        epoch + 1, train_acc_metric, train_auc_result
                    )
                )
                writer.add_scalars('AUC', {"train_auc": train_auc_result}, epoch + 1)
                writer.add_scalars('Accuracy', {"train_accuracy": train_acc_metric}, epoch + 1)

                # Val set evaluation
                val_y_pred = torch.tensor([], dtype=torch.float32, device=device)
                val_y = torch.tensor([], dtype=torch.float32, device=device)
                val_loss_total = 0
                for val_data in val_loader:
                    val_imgA, val_imgB, val_labels = val_data['imgA'].float().to(device), \
                        val_data['imgB'].float().to(device), \
                        val_data['label'].float().to(device)
                    val_output = net(val_imgA, val_imgB).squeeze()
                    val_output_prob = torch.sigmoid(val_output)
                    val_y_pred = torch.cat([val_y_pred, val_output_prob], dim=0)
                    val_y = torch.cat([val_y, val_labels], dim=0)
                    val_loss = loss_function(val_output, val_labels)
                    val_loss_total += val_loss.item()

                val_loss_avg = val_loss_total / len(val_loader)
                auto_scheduler.step(val_loss_avg)
                step_lr_list.append(auto_scheduler.get_last_lr()[0])
                writer.add_scalars('Loss', {"average_val_loss": val_loss_avg}, epoch + 1)
                custom_print(f"epoch {epoch + 1} average_val loss: {val_loss_avg:.4f}")

                val_y_pred_np = val_y_pred.cpu().numpy()
                val_y_np = val_y.cpu().numpy()
                val_acc_metric = accuracy_score(val_y_np, val_y_pred_np > tresh_hold)
                val_auc_result = roc_auc_score(val_y_np, val_y_pred_np)

                combined_metric = 4 * val_auc_result + val_acc_metric
                if epoch + 1 >= 100:
                    if combined_metric > best_metric_100:
                        best_metric_100 = combined_metric
                        best_metric_epoch_100 = epoch + 1
                        torch.save(net.state_dict(), f"{save_path}/Best_Model_After100.pth")
                        custom_print(
                            f"saved_100 new Best metric model after 100 epo at epoch {epoch + 1} with Accuracy {val_acc_metric:.4f} and AUC {val_auc_result:.4f}")
                if val_auc_result > best_auc:
                    best_auc = val_auc_result
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1


                custom_print(
                    "current epoch: {} val accuracy: {:.4f} val AUC: {:.4f}".format(
                        epoch + 1, val_acc_metric, val_auc_result
                    )
                )
                writer.add_scalars('AUC', {"val_auc": val_auc_result}, epoch + 1)
                writer.add_scalars('Accuracy', {"val_accuracy": val_acc_metric}, epoch + 1)

    custom_print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    custom_print(f"train completed, best_metric_after100: {best_metric_100:.4f} at epoch: {best_metric_epoch_100}")
    writer.close()
    save_logs_to_json(filename=logjson_savepath)


if __name__ == '__main__':
    main()
