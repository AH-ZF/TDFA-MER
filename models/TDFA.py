import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

import cbam
from config import args



device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')


class TDFA(nn.Module):
    def __init__(self, size):
        super(TDFA, self).__init__()
        self.feature_num = 64
        self.size = size  # 10
        self.allsize = 60
        self.net_struct = {
            'conv2_0': {8: 7200, 10: 4608, 20: 1152, 30: 512, 40: 288},
            'conv2_1': {8: 512, 10: 512, 20: 256, 30: 128, 40: 32},
            'fc1': {8: 4608, 10: 8192, 20: 12544, 30: 12800, 40: 6272},
        }

        self.pool = nn.MaxPool2d(3, 3, 1)

        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        if args.isallfeature:
            self.conv2 = nn.Conv2d(4736, self.net_struct['conv2_1'][self.size], 1)
            if args.isallfeaStruct == 11:
                self.conv3 = nn.Conv2d(3, 8, 3, 3, 0)
                self.pool3 = nn.MaxPool2d(5, 5, 0)
                pass
            if args.isallfeaStruct == 23:
                self.conv4 = nn.Conv2d(3, 5, 3, 1, 1)
                self.pool4 = nn.MaxPool2d(3, 3, 0)
                self.pool5 = nn.MaxPool2d(5, 5, 0)
                self.conv5 = nn.Conv2d(5, 8, 3, 1, 1)
                pass

            pass
        else:
            self.conv2 = nn.Conv2d(self.net_struct['conv2_0'][self.size], self.net_struct['conv2_1'][self.size], 1)
            pass


        self.fc1 = nn.Linear(self.net_struct['fc1'][self.size], 1024)
        self.fc2 = nn.Linear(1024, self.feature_num)
        # Consider CBAM modules
        self.cbams = cbam.CBAM(gate_channels=3, reduction_ratio=2, no_spatial=True).to(device)
        self.dps = nn.Dropout(0.35)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=self.net_struct['conv2_1'][self.size])

        # self.cbam = CBAM(512)

    def forward(self, input1, input2, input3, input4):

        if args.isallfeature:
            blocksall1 = []
            allconv1 = []
            na = 120 // self.allsize
            # Sub-region size 10*10
            for i in range(na):
                for j in range(na):
                    blocksall1.append(
                        input1[:, :, self.allsize * i:self.allsize * (i + 1), self.allsize * j:self.allsize * (j + 1)])
                pass
            for i in range(na ** 2):
                if args.isallfeaStruct == 11:
                    allconv1.append(self.pool3(F.relu(self.conv3(blocksall1[i]))))
                    pass
                if args.isallfeaStruct == 23:
                    midblock = self.pool4(F.relu(self.conv4(blocksall1[i])))
                    allconv1.append(self.pool5(F.relu(self.conv5(midblock))))
                    pass
                pass
            allx1 = torch.cat((allconv1[0], allconv1[1]), dim=1)
            for i in range(2, na ** 2):
                allx1 = torch.cat((allx1, allconv1[i]), dim=1)
                pass
            blocksall2 = []
            allconv2 = []
            for i in range(na):
                for j in range(na):
                    blocksall2.append(
                        input2[:, :, self.allsize * i:self.allsize * (i + 1), self.allsize * j:self.allsize * (j + 1)])
                pass
            for i in range(na ** 2):
                if args.isallfeaStruct == 11:
                    allconv2.append(self.pool3(F.relu(self.conv3(blocksall2[i]))))
                    pass
                if args.isallfeaStruct == 23:
                    midblock = self.pool4(F.relu(self.conv4(blocksall2[i])))
                    allconv2.append(self.pool5(F.relu(self.conv5(midblock))))
                    pass
                pass
            allx2 = torch.cat((allconv2[0], allconv2[1]), dim=1)
            for i in range(2, na ** 2):
                allx2 = torch.cat((allx2, allconv2[i]), dim=1)
                pass
            blocksall3 = []
            allconv3 = []
            for i in range(na):
                for j in range(na):
                    blocksall3.append(
                        input3[:, :, self.allsize * i:self.allsize * (i + 1), self.allsize * j:self.allsize * (j + 1)])
                pass
            for i in range(na ** 2):

                if args.isallfeaStruct == 11:
                    allconv3.append(self.pool3(F.relu(self.conv3(blocksall3[i]))))
                    pass
                if args.isallfeaStruct == 23:
                    midblock = self.pool4(F.relu(self.conv4(blocksall3[i])))
                    allconv3.append(self.pool5(F.relu(self.conv5(midblock))))
                    pass
                pass
            allx3 = torch.cat((allconv3[0], allconv3[1]), dim=1)
            for i in range(2, na ** 2):
                allx3 = torch.cat((allx3, allconv3[i]), dim=1)
                pass
            blocksall4 = []
            allconv4 = []
            for i in range(na):
                for j in range(na):
                    blocksall4.append(
                        input4[:, :, self.allsize * i:self.allsize * (i + 1), self.allsize * j:self.allsize * (j + 1)])
                pass
            for i in range(na ** 2):

                if args.isallfeaStruct == 11:
                    allconv4.append(self.pool3(F.relu(self.conv3(blocksall4[i]))))
                    pass
                if args.isallfeaStruct == 23:
                    midblock = self.pool4(F.relu(self.conv4(blocksall4[i])))
                    allconv4.append(self.pool5(F.relu(self.conv5(midblock))))
                    pass
                pass
            allx4 = torch.cat((allconv4[0], allconv4[1]), dim=1)
            for i in range(2, na ** 2):
                allx4 = torch.cat((allx4, allconv4[i]), dim=1)
                pass
            pass
        # 144 sub-region images of a single frame saved
        blocks1 = []
        # size refers to the size of the subregion of a single frame.
        n = 120 // self.size
        mu1 = torch.ones((input1.size(0), n ** 2)).to(device)
        mu2 = mu1
        mu3 = mu1
        mu4 = mu1
        scal = 1.0
        # Construct CBAM feature augmentation
        if args.ram:
            mu1 = self.cbams(input1).to(device)
            mu2 = self.cbams(input2).to(device)
            mu3 = self.cbams(input3).to(device)
            mu4 = self.cbams(input4).to(device)

        # Subregion size 10*10
        for i in range(n):
            for j in range(n):
                blocks1.append(input1[:, :, self.size * i:self.size * (i + 1), self.size * j:self.size * (j + 1)])

        convs1 = []
        for i in range(n ** 2):
            convs1.append(self.pool(F.relu(self.conv1(blocks1[i]))))

        x1 = torch.cat((mu1[:, 0].view(mu1.size(0), 1, 1, 1).expand_as(convs1[0]) * convs1[0],
                        mu1[:, 1].view(mu1.size(0), 1, 1, 1).expand_as(convs1[1]) * convs1[1]), dim=1)

        for i in range(2, n ** 2):
            x1 = torch.cat((x1, mu1[:, i].view(mu1.size(0), 1, 1, 1).expand_as(convs1[i]) * convs1[i]), dim=1)


        blocks2 = []
        for i in range(n):
            for j in range(n):
                blocks2.append(input2[:, :, self.size * i:self.size * (i + 1), self.size * j:self.size * (j + 1)])

        convs2 = []
        for i in range(n ** 2):
            convs2.append(self.pool(F.relu(self.conv1(blocks2[i]))))

        x2 = torch.cat((mu2[:, 0].view(mu2.size(0), 1, 1, 1).expand_as(convs2[0]) * convs2[0],
                        mu2[:, 1].view(mu2.size(0), 1, 1, 1).expand_as(convs2[1]) * convs2[1]), dim=1)
        for i in range(2, n ** 2):
            x2 = torch.cat((x2, mu2[:, i].view(mu2.size(0), 1, 1, 1).expand_as(convs2[i]) * convs2[i]), dim=1)

        blocks3 = []
        for i in range(n):
            for j in range(n):
                blocks3.append(input3[:, :, self.size * i:self.size * (i + 1), self.size * j:self.size * (j + 1)])

        convs3 = []
        for i in range(n ** 2):
            convs3.append(self.pool(F.relu(self.conv1(blocks3[i]))))

        x3 = torch.cat((mu3[:, 0].view(mu3.size(0), 1, 1, 1).expand_as(convs3[0]) * convs3[0],
                        mu3[:, 1].view(mu3.size(0), 1, 1, 1).expand_as(convs3[1]) * convs3[1]), dim=1)
        for i in range(2, n ** 2):
            x3 = torch.cat((x3, mu3[:, i].view(mu3.size(0), 1, 1, 1).expand_as(convs3[i]) * convs3[i]), dim=1)

        blocks4 = []
        for i in range(n):
            for j in range(n):
                blocks4.append(input4[:, :, self.size * i:self.size * (i + 1), self.size * j:self.size * (j + 1)])
        convs4 = []
        for i in range(n ** 2):

            convs4.append(self.pool(F.relu(self.conv1(blocks4[i]))))

        x4 = torch.cat((mu4[:, 0].view(mu4.size(0), 1, 1, 1).expand_as(convs4[0]) * convs4[0],
                        mu4[:, 1].view(mu4.size(0), 1, 1, 1).expand_as(convs4[1]) * convs4[1]), dim=1)
        for i in range(2, n ** 2):
            x4 = torch.cat((x4, mu4[:, i].view(mu4.size(0), 1, 1, 1).expand_as(convs4[i]) * convs4[i]), dim=1)


        allfeaW = 1.0
        allxiW = 1.0
        # Weighted splicing fusion processing
        if args.isallfeature:
            x1 = torch.cat((allxiW*x1, allx1 * allfeaW), dim=1)
            x2 = torch.cat((allxiW*x2, allx2 * allfeaW), dim=1)
            x3 = torch.cat((allxiW*x3, allx3 * allfeaW), dim=1)
            x4 = torch.cat((allxiW*x4, allx4 * allfeaW), dim=1)
            pass

        # Dimensional splicing
        x = torch.cat((x1, x2, x3, x4), dim=1) 


        x = F.relu(self.conv2(x)) 

        x = x.view(x.size(0), -1)  

        x = F.relu(self.fc1(self.dps(x)))
        x = F.relu(self.fc2(x))  


        return x
