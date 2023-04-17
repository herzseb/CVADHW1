import torch.nn as nn
import torch
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features
        self.queue_length = 10
        self.input_size = 512
        self.hidden_size = 256
        self.num_layers = 2
        self.drop_out = 0.2
        self.in_channel = 10
        self.out_channels = 1
        self.memory_size = 10
        self.do = nn.Dropout(self.drop_out)
        self.grus = nn.ModuleList()
        self.fc_features =  nn.Linear(512 * 7 * 7, self.input_size)
        for i in range(7):
            self.grus.append(torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                                          num_layers=self.num_layers, batch_first=True, dropout=self.drop_out))

        self.batchnorm = torch.nn.BatchNorm1d(10)
        self.blocks = nn.ModuleList()
        for i in range(7):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(self.memory_size),
                nn.ReLU(),
                nn.Dropout(self.drop_out),
                torch.nn.Conv1d(in_channels=self.in_channel,
                                out_channels=self.out_channels, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 1),
            ))

        self.traffic_light_state = nn.Sequential(
            nn.Linear( self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size, 3),
        )

    def forward(self, img, command, memory_stack):
        features = self.feature_extractor(img)
        features = features.view(-1, 512 * 7 * 7)
        features = torch.relu(features)
        features = self.fc_features(features)
        features = torch.unsqueeze(features, dim=1)
        self.percep_memory = torch.cat((features, memory_stack), dim=1)

        self.percep_memory = torch.relu(self.percep_memory)
        self.percep_memory = self.do(self.percep_memory)

        if command[0] == 0:
            affordance_lane_dist = self.blocks[0](
                self.grus[0](self.percep_memory)[0])
        elif command[0] == 1:
            affordance_lane_dist = self.blocks[1](
                self.grus[1](self.percep_memory)[0])
        elif command[0] == 2 or command[0] == 3:
            affordance_lane_dist = self.blocks[2](self.grus[2](self.percep_memory)[0])

        if command[0] == 0:
            affordance_angle = self.blocks[3](
                self.grus[3](self.percep_memory)[0])
        elif command[0] == 1:
            affordance_angle = self.blocks[4](
                self.grus[4](self.percep_memory)[0])
        elif command[0] == 2 or command[0] == 3:
            affordance_angle = self.blocks[5](
                self.grus[5](self.percep_memory)[0])

        affordance_traffic_light_distance = self.blocks[6](
            self.grus[6](self.percep_memory)[0])

        affordance_traffic_light_state = self.traffic_light_state(torch.squeeze(features, dim=1))
        features_to_memory = features.detach()
        return [torch.squeeze(torch.concat((affordance_lane_dist, affordance_angle, affordance_traffic_light_distance), dim=1),dim=2), affordance_traffic_light_state], features_to_memory
