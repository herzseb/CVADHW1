import torch.nn as nn
import torch


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.train()
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
        self.after_concat = nn.Linear(1024, 512)
        self.action_straight = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,2),
        )
        self.action_left = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,2),
        )
        self.action_right = nn.Sequential(
          nn.Linear(512,512),
          nn.ReLU(),
          nn.Linear(512,2),
        )


    def forward(self, img, command, measured_speed):
        img_enc = self.model(img)
        speed_enc = self.speed_encoding(measured_speed)
        embedding = self.after_concat(torch.cat((img_enc, speed_enc), dim=1))
        if command == "STRAIGHT":
            return self.action_straight(embedding)
        elif command == "LEFT":
            return self.action_left(embedding)
        elif command == "RIGHT":
            return self.action_right(embedding)
        else:
            raise RuntimeError (f"Command has unexpected value {command}")
