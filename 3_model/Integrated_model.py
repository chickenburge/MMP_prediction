import torch
import torch.nn as nn
import torch.nn.init as init
from 3_model.resnet_models import resnet10


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = x * attention_weights
        weighted_output = torch.sum(weighted_output, dim=1)
        return weighted_output

class DualTowerResNet10(nn.Module):
    def __init__(self, num_class=2):
        super(DualTowerResNet10, self).__init__()
        self.tower1 = resnet10(num_classes=256)
        self.tower2 = resnet10(num_classes=256)
        self.mlp = nn.Sequential(
            nn.Linear(256 * 2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.attention = Attention(input_dim=32 + 4, attention_dim=16)
        self.classifier = nn.Sequential(
            nn.Linear(32 + 4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4, num_class)
        )

    def forward(self, img1, img2, wbc, ne, d_d, lactic):
        out1 = self.tower1(img1)
        out2 = self.tower2(img2)
        out = torch.cat((out1, out2), dim=1)
        img_feature = self.mlp(out)
        clinical_feature = torch.stack([wbc, ne, d_d, lactic], dim=1).to(out.device)
        combined_features = torch.cat((img_feature, clinical_feature), dim=1).unsqueeze(1)
        attended_features = self.attention(combined_features)
        out = self.classifier(attended_features)
        return out

    def load_pretrained_weights(self, weights_path1, weights_path2):
        weights1 = torch.load(weights_path1)
        weights1 = weights1['state_dict']
        weights1 = {k.replace("module.", ""): v for k, v in weights1.items()}
        model_dict1 = self.tower1.state_dict()
        model_dict1.update(weights1)
        self.tower1.load_state_dict(model_dict1)
        weights2 = torch.load(weights_path2)
        weights2 = weights2['state_dict']
        weights2 = {k.replace("module.", ""): v for k, v in weights2.items()}
        model_dict2 = self.tower2.state_dict()
        model_dict2.update(weights2)
        self.tower2.load_state_dict(model_dict2)
        self.initialize_mlp_weights()

    def initialize_mlp_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)







