import torch.nn as nn
import torch
import torchvision.models as models


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        # self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.resnet.eval()
        # resnet = models.resnet18(pretrained=True)
        # modules=list(resnet.children())[:-1]
        # self.resnet=nn.Sequential(*modules)
        # for p in resnet.parameters():
        #     p.requires_grad = False
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Parameters of newly constructed modules have requires_grad=True by default
        self.resnet_out = 512
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.resnet_out)

        self.dropout = 0.5 #0.2
        self.hidden = 128
        self.hidden_speed_prediction = 64
        self.speed_features = 64
        self.feature_input = self.resnet_out + self.speed_features
        self.feature_output = 512
        self.speed_encoding = nn.Sequential(
          nn.Linear(1, self.speed_features),
          #nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.speed_features,self.speed_features),
        )
        self.speed_prediction = nn.Sequential(
          nn.ReLU(),
          nn.Linear(self.resnet_out,self.hidden_speed_prediction),
          #nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Linear(self.hidden_speed_prediction,self.hidden_speed_prediction),
          #nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden_speed_prediction,1),
        )
        self.after_concat = nn.Sequential(
          #nn.BatchNorm1d(1128),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.feature_input,self.feature_output),
          #nn.BatchNorm1d(1128),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          # nn.Linear(self.blocks_input,self.feature_output),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          # nn.Dropout(p=self.dropout),
        )
        self.action_straight = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(self.feature_output,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Linear(self.hidden,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_left = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(self.feature_output,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Linear(self.hidden,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_right = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(self.feature_output,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Linear(self.hidden,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_follow = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(self.feature_output,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Linear(self.hidden,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )


    def forward(self, img, command, measured_speed):
        img_enc = self.resnet(img)
        speed_enc = self.speed_encoding(measured_speed)
        embedding = self.after_concat(torch.cat((img_enc, speed_enc), dim=1))
        if command[0] == 0: #LEFT
            return torch.concat((self.action_left(embedding), self.speed_prediction(img_enc)), dim=1)
        elif command[0] == 1: #RIGHT
            return torch.concat((self.action_right(embedding), self.speed_prediction(img_enc)), dim=1)
        elif command[0] == 2: #STRAIGHT
            return torch.concat((self.action_straight(embedding), self.speed_prediction(img_enc)), dim=1)
        elif command[0] == 3: #LANEFOLLOW
          return torch.concat((self.action_follow(embedding), self.speed_prediction(img_enc)), dim=1)
        else:
            raise RuntimeError (f"Command has unexpected value {command}")
