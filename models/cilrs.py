import torch.nn as nn
import torch


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet.train()
        self.speed_encoding = nn.Sequential(
          nn.Linear(1,512),
          nn.ReLU(),
          nn.Linear(512,512),
        )
        self.speed_prediction = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,1),
        )
        self.after_concat = nn.Linear(1512, 512)
        self.action_straight = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,3),
        )
        self.action_left = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,3),
        )
        self.action_right = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,3),
        )
        self.action_follow = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,3),
        )


    def forward(self, img, command, measured_speed):
        img_enc = self.resnet(img)
        speed_enc = self.speed_encoding(measured_speed)
        embedding = self.after_concat(torch.cat((img_enc, speed_enc), dim=1))
        if command[0] == 0: #LEFT
            return torch.concat((self.action_left(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 1: #RIGHT
            return torch.concat((self.action_left(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 2: #STRAIGHT
            return torch.concat((self.action_left(embedding), self.speed_prediction(embedding)), dim=1)
        elif command[0] == 3: #LANEFOLLOW
          return torch.concat((self.action_left(embedding), self.speed_prediction(embedding)), dim=1)
        else:
            raise RuntimeError (f"Command has unexpected value {command}")