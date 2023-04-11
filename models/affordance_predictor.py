import torch.nn as nn
import torch
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.train()
        self.percep_memory = torch.zeros(2, 9, 1000)
        self.percep_memory = self.percep_memory.to(device)
        self.queue_length = 10
        self.input_size = 1000
        self.hidden_size = 512
        self.num_layers = 3
        self.drop_out = 0.2
        self.in_channel = 10
        self.out_channels = 1
        self.do = nn.Dropout(self.drop_out)
        self.grus = nn.ModuleList()
        for i in range(7):
            self.grus.append(torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                                          num_layers=self.num_layers, batch_first=True, dropout=self.drop_out))

        # lane_dist_left_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                                   num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # lane_dist_right_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                                    num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # lane_dist_straight_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                                       num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # angle_left_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                               num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # angle_right_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                                num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # angle_straight_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                                   num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        # tl_state_gru = torch.nn.GRU(input_size=self.input_size, hidden_layers=self.hidden_size,
        #                             num_layers=self.num_layers, batch_first=True, dropout=self.drop_out)
        self.batchnorm = torch.nn.BatchNorm1d(10)
        self.blocks = nn.ModuleList()
        for i in range(7):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Dropout(self.drop_out),
                torch.nn.Conv1d(in_channels=self.in_channel,
                                out_channels=self.out_channels, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 1),
            ))
        # self.lane_dist_left = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     torch.nn.Conv1d(in_channels=self.in_channel,
        #                     out_channels=self.out_channels, kernel_size=3, padding="same"),
        #     nn.BatchNorm1d(self.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )
        # self.lane_dist_right = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )
        # self.lane_dist_straight = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )

        # self.angle_left = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )
        # self.angle_right = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )
        # self.angle_straight = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )

        # self.traffic_light_distance = nn.Sequential(
        #     nn.BatchNorm1d(10),
        #     nn.ReLU(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        # )

        self.traffic_light_state = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(512, 3),
        )

    def forward(self, img, command):
        features = self.feature_extractor(img)
        features = torch.unsqueeze(features, dim=1)
        self.percep_memory = torch.concat(
            (features, self.percep_memory), dim=1)
        self.percep_memory = self.batchnorm(self.percep_memory)
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

        affordance_traffic_light_state = self.traffic_light_state(
            self.percep_memory[:, 0, :])
        self.percep_memory = self.percep_memory[:, :10, :]
        return [torch.squeeze(torch.concat((affordance_lane_dist, affordance_angle, affordance_traffic_light_distance), dim=1),dim=2), affordance_traffic_light_state]
