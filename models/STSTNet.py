import torch.nn as nn
import torch

class STSTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(STSTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels=5, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(5)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=5 * 5 * 16, out_features=out_channels)

    def forward(self, x):
        # x=
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.maxpool(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        x2 = self.maxpool(x2)
        x2 = self.dropout(x2)

        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.bn3(x3)
        x3 = self.maxpool(x3)
        x3 = self.dropout(x3)

        x = torch.cat((x1, x2, x3), 1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x