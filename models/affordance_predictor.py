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
        self.speed_encoding = nn.Sequential(
          nn.Linear(1,512),
          nn.ReLU(),
          nn.Linear(512,512),
        )

    def forward(self, img):
        features = self.feature_extractor(img)
        self.percep_memory.push(features)
        features_lane_dist = torch.concat((self.percep_memory[:11]), dim=1) #Check 
        
        
        lane distance (conditional)
        
        route angle (conditional)
        
        traffic light distance
        
        traffic light state
        
