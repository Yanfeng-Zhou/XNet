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

class WDS(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(WDS, self).__init__()

        # branch1
        self.b1_1 = basic_block(in_channels, 64)
        self.b1_2 = basic_block(64, 64)
        self.b1_3 = basic_block(64, 64)
        self.b1_4 = basic_block(64, 64)
        self.b1_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.b1_6 = basic_block(64, 128)
        self.b1_7 = basic_block(128, 128)
        self.b1_8 = basic_block(128, 128)
        self.b1_9 = basic_block(128, 128)
        self.b1_10 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # branch2
        self.b2_1 = basic_block(in_channels, 64)
        self.b2_2 = basic_block(64, 64)
        self.b2_3 = basic_block(64, 64)
        self.b2_4 = basic_block(64, 64)
        self.b2_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.b2_6 = basic_block(64, 128)
        self.b2_7 = basic_block(128, 128)
        self.b2_8 = basic_block(128, 128)
        self.b2_9 = basic_block(128, 128)
        self.b2_10 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # branch3
        self.b3_1 = basic_block(in_channels, 64)
        self.b3_2 = basic_block(64, 64)
        self.b3_3 = basic_block(64, 64)
        self.b3_4 = basic_block(64, 64)
        self.b3_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.b3_6 = basic_block(64, 128)
        self.b3_7 = basic_block(128, 128)
        self.b3_8 = basic_block(128, 128)
        self.b3_9 = basic_block(128, 128)
        self.b3_10 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # branch4
        self.b4_1 = basic_block(in_channels, 64)
        self.b4_2 = basic_block(64, 64)
        self.b4_3 = basic_block(64, 64)
        self.b4_4 = basic_block(64, 64)
        self.b4_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.b4_6 = basic_block(64, 128)
        self.b4_7 = basic_block(128, 128)
        self.b4_8 = basic_block(128, 128)
        self.b4_9 = basic_block(128, 128)
        self.b4_10 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # output
        self.output_layer = nn.Sequential(
            nn.Conv2d(128*4, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
        )

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

    def forward(self, LL, LH, HL, HH):

        # H, W = 2*LL.shape[2], 2*LL.shape[3]
        H, W = LL.shape[2], LL.shape[3]

        LL = self.b1_1(LL)
        LL = self.b1_2(LL)
        LL = self.b1_3(LL)
        LL = self.b1_4(LL)
        LL = self.b1_5(LL)
        LL = self.b1_6(LL)
        LL = self.b1_7(LL)
        LL = self.b1_8(LL)
        LL = self.b1_9(LL)
        LL = self.b1_10(LL)

        LH = self.b2_1(LH)
        LH = self.b2_2(LH)
        LH = self.b2_3(LH)
        LH = self.b2_4(LH)
        LH = self.b2_5(LH)
        LH = self.b2_6(LH)
        LH = self.b2_7(LH)
        LH = self.b2_8(LH)
        LH = self.b2_9(LH)
        LH = self.b2_10(LH)

        HL = self.b3_1(HL)
        HL = self.b3_2(HL)
        HL = self.b3_3(HL)
        HL = self.b3_4(HL)
        HL = self.b3_5(HL)
        HL = self.b3_6(HL)
        HL = self.b3_7(HL)
        HL = self.b3_8(HL)
        HL = self.b3_9(HL)
        HL = self.b3_10(HL)

        HH = self.b4_1(HH)
        HH = self.b4_2(HH)
        HH = self.b4_3(HH)
        HH = self.b4_4(HH)
        HH = self.b4_5(HH)
        HH = self.b4_6(HH)
        HH = self.b4_7(HH)
        HH = self.b4_8(HH)
        HH = self.b4_9(HH)
        HH = self.b4_10(HH)

        x = torch.cat((LL, LH, HL, HH), dim=1)
        x = self.output_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    from loss.loss_function import segmentation_loss
    criterion = segmentation_loss('dice', False)
    mask = torch.ones(2, 128, 128).long()
    model = WDS(1, 5)
    model.train()
    input1 = torch.rand(2, 1, 128, 128)
    y = model(input1, input1, input1, input1)
    loss_train = criterion(y, mask)
    loss_train.backward()
    # print(output)
    print(y.data.cpu().numpy().shape)
    print(loss_train)
