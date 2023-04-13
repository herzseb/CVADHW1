import torch.nn as nn
import torch


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.eval()
        self.dropout = 0.2 #0.2
        self.hidden = 128
        self.speed_encoding = nn.Sequential(
          nn.Linear(1,64),
          #nn.BatchNorm1d(64),
          # nn.ReLU(),
          # nn.Dropout(p=self.dropout),
          # nn.Linear(64,128),
          # nn.ReLU()
        )
        self.speed_prediction = nn.Sequential(
          nn.Linear(512,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,1),
        )
        self.after_concat = nn.Sequential(
          #nn.BatchNorm1d(1128),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(1064,1064),
          #nn.BatchNorm1d(1128),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(1064,512),
          #nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
        )
        self.action_straight = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(512,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_left = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(512,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_right = nn.Sequential(
          # nn.Linear(512,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(512,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )
        self.action_follow = nn.Sequential(
          # nn.Linear(521,512),
          # #nn.BatchNorm1d(512),
          # nn.ReLU(),
          nn.Linear(512,self.hidden),
          #nn.BatchNorm1d(self.hidden),
          nn.ReLU(),
          nn.Dropout(p=self.dropout),
          nn.Linear(self.hidden,3)
        )


    def forward(self, img, command, measured_speed):
        img_enc = self.resnet(img)
        #speed_enc = self.speed_encoding(measured_speed)
        embedding = self.after_concat(torch.cat((img_enc, measured_speed), dim=1))
        if command[0] == 0: #LEFT
            return torch.concat((self.action_left(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 1: #RIGHT
            return torch.concat((self.action_right(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 2: #STRAIGHT
            return torch.concat((self.action_straight(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 3: #LANEFOLLOW
          return torch.concat((self.action_follow(embedding), self.speed_prediction(embedding)), dim=1)
        else:
            raise RuntimeError (f"Command has unexpected value {command}")
