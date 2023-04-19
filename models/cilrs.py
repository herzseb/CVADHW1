import torch.nn as nn
import torch
import torchvision.models as models


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""

    def __init__(self):
        super(CILRS, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet_out = 512
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.resnet_out)

        self.dropout = 0.3  # 0.2
        self.hidden = 256
        self.hidden_speed_prediction = 64
        self.speed_features = 64
        self.feature_input = self.resnet_out + self.speed_features
        self.feature_output = 512
        self.speed_encoding = nn.Sequential(
            nn.Linear(1, self.speed_features),
        )
        self.speed_prediction = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.resnet_out, self.hidden_speed_prediction),
            nn.BatchNorm1d(self.hidden_speed_prediction),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_speed_prediction, 1),
        )
        self.after_concat = nn.Sequential(
            nn.BatchNorm1d(self.feature_input),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_input, self.feature_output),
            nn.BatchNorm1d(self.feature_output),
            nn.LeakyReLU(),
        )
        action_block = []
        for i in range(4):
            action_block.append(nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.feature_output, self.hidden),
                nn.BatchNorm1d(self.hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden, 3)
            ))

    def forward(self, img, command, measured_speed):
        img_enc = self.resnet(img)
        speed_enc = self.speed_encoding(measured_speed)
        embedding = self.after_concat(torch.cat((img_enc, speed_enc), dim=1))
        return torch.concat((self.action_block[command[0]](embedding), self.speed_prediction(img_enc)), dim=1)
