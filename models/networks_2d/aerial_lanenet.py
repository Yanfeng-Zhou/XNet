import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np


class basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.block(x)
        return x

class Aerial_LaneNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Aerial_LaneNet, self).__init__()

        l1, l2, l3, l4, l5 = 64, 128, 256, 512, 512
        dropout = 0.2

        # e1
        self.conv1_1 = basic_block(in_channels, l1)
        self.conv1_2 = basic_block(l1, l1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # e2
        self.conv2_1 = basic_block(l1+3, l2)
        self.conv2_2 = basic_block(l2, l2)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # e3
        self.conv3_1 = basic_block(l2+3, l3)
        self.conv3_2 = basic_block(l3, l3)
        self.conv3_3 = basic_block(l3, l3)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # e4
        self.conv4_1 = basic_block(l3+3, l4)
        self.conv4_2 = basic_block(l4, l4)
        self.conv4_3 = basic_block(l4, l4)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # e5
        self.conv5_1 = basic_block(l4+3, l5)
        self.conv5_2 = basic_block(l5, l5)
        self.conv5_3 = basic_block(l5, l5)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # e6
        self.conv6_1 = basic_block(l5, 4096)
        self.drop6_1 = nn.Dropout2d(dropout)
        self.conv6_2 = basic_block(4096, 4096)
        self.drop6_2 = nn.Dropout2d(dropout)
        self.conv6_3 = nn.ConvTranspose2d(4096, l5, kernel_size=4, stride=2, padding=1, bias=False)

        # d4
        self.conv4_4 = basic_block(2*l5, l5)
        self.drop4_4 = nn.Dropout2d(dropout)
        self.conv4_5 = nn.ConvTranspose2d(l5, l3, kernel_size=4, stride=2, padding=1, bias=False)

        # d3
        self.conv3_4 = basic_block(2*l3, l3)
        self.drop3_4 = nn.Dropout2d(dropout)
        self.conv3_5 = nn.ConvTranspose2d(l3, l2, kernel_size=4, stride=2, padding=1, bias=False)

        # d2
        self.conv2_4 = basic_block(2*l2, l2)
        self.drop2_4 = nn.Dropout2d(dropout)
        self.conv2_5 = nn.ConvTranspose2d(l2, l1, kernel_size=4, stride=2, padding=1, bias=False)

        # d1
        self.conv1_3 = basic_block(2*l1, l1)
        self.drop1_3 = nn.Dropout2d(dropout)
        self.conv1_4 = nn.ConvTranspose2d(l1, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, x_wavelet_1, x_wavelet_2, x_wavelet_3, x_wavelet_4):

        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.pool1(x1)

        x2 = torch.cat((x1, x_wavelet_1), dim=1)
        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.pool2(x2)

        x3 = torch.cat((x2, x_wavelet_2), dim=1)
        x3 = self.conv3_1(x3)
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)
        x3 = self.pool3(x3)

        x4 = torch.cat((x3, x_wavelet_3), dim=1)
        x4 = self.conv4_1(x4)
        x4 = self.conv4_2(x4)
        x4 = self.conv4_3(x4)
        x4 = self.pool4(x4)

        x5 = torch.cat((x4, x_wavelet_4), dim=1)
        x5 = self.conv5_1(x5)
        x5 = self.conv5_2(x5)
        x5 = self.conv5_3(x5)
        x5 = self.pool5(x5)

        x6 = self.conv6_1(x5)
        x6 = self.drop6_1(x6)
        x6 = self.conv6_2(x6)
        x6 = self.drop6_2(x6)
        x6 = self.conv6_3(x6)

        x5 = torch.cat((x6, x4), dim=1)
        x5 = self.conv4_4(x5)
        x5 = self.drop4_4(x5)
        x5 = self.conv4_5(x5)

        x4 = torch.cat((x5, x3), dim=1)
        x4 = self.conv3_4(x4)
        x4 = self.drop3_4(x4)
        x4 = self.conv3_5(x4)

        x3 = torch.cat((x4, x2), dim=1)
        x3 = self.conv2_4(x3)
        x3 = self.drop2_4(x3)
        x3 = self.conv2_5(x3)

        x2 = torch.cat((x3, x1), dim=1)
        x2 = self.conv1_3(x2)
        x2 = self.drop1_3(x2)
        x2 = self.conv1_4(x2)

        return x2

# if __name__ == '__main__':
#     from loss.loss_function import segmentation_loss
#     criterion = segmentation_loss('dice', False)
#     mask = torch.ones(2, 128, 128).long()
#     model = Aerial_LaneNet(1, 5)
#     model.train()
#     input1 = torch.rand(2, 1, 128, 128)
#     input2 = torch.rand(2, 3, 64, 64)
#     input3 = torch.rand(2, 3, 32, 32)
#     input4 = torch.rand(2, 3, 16, 16)
#     input5 = torch.rand(2, 3, 8, 8)
#
#     y = model(input1, input2, input3, input4, input5)
#     loss_train = criterion(y, mask)
#     loss_train.backward()
#     # print(output)
#     print(y.data.cpu().numpy().shape)
#     print(loss_train)