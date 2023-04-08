import torch.nn as nn
import torch
import torchvision.models as models


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.train()
        self.percep_memory = []
        self.queue_length = 10
        self.lane_dist_left = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.lane_dist_right = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.lane_dist_straight = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.angle_left = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.angle_right = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.angle_straight = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.traffic_light_distance = nn.GRU(10, 1000, 2)

        self.traffic_light_state = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, img, command):
        features = self.feature_extractor(img)
        self.percep_memory.append(torch.unsqueeze(features,dim=0))
        features_lane_dist = torch.concat(tuple(self.percep_memory[-self.queue_length:]), dim=1)  # Check
        features_angle = torch.concat(tuple(self.percep_memory[-self.queue_length:]), dim=1)  # Check
        features_traffic_light_distance = torch.concat(tuple(self.percep_memory[-self.queue_length:]), dim=0)  # Check
        features_traffic_light_state = torch.concat(tuple(self.percep_memory[-self.queue_length:]), dim=1)  # Check

        if command[0] == 0:
            affordance_lane_dist = self.lane_dist_left(features_lane_dist)
        elif command[0] == 1:
            affordance_lane_dist = self.lane_dist_right(features_lane_dist)
        elif command[0] == 2 or command[0] == 3:
            affordance_lane_dist = self.lane_dist_straight(features_lane_dist)

        if command[0] == 0:
            affordance_angle = self.angle_left(features_angle)
        elif command[0] == 1:
            affordance_angle = self.angle_right(features_angle)
        elif command[0] == 2 or command[0] == 3:
            affordance_angle = self.angle_straight(features_angle)

        affordance_traffic_light_distance = self.traffic_light_distance(
            features_traffic_light_distance)

        affordance_traffic_light_state = self.traffic_light_state(
            features_traffic_light_state)
        self.percep_memory.pop(0)
        return [torch.concat((affordance_lane_dist, affordance_angle, affordance_traffic_light_distance), dim=1), affordance_traffic_light_state]
