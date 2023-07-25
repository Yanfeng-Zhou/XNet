import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
BatchNorm2d = nn.BatchNorm2d
relu_inplace = True

BN_MOMENTUM = 0.1
# BN_MOMENTUM = 0.01


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class same_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(same_conv, self).__init__()
        self.same = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace))
    def forward(self, x):
        x = self.same(x)
        return x

class transition_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(transition_conv, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(ch_out, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace))
    def forward(self, x):
        x = self.transition(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn2(out) + identity
        out = self.relu(out)

        return out

class DoubleBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(DoubleBasicBlock, self).__init__()

        self.DBB = nn.Sequential(
            BasicBlock(inplanes=inplanes, planes=planes, downsample=downsample),
            BasicBlock(inplanes=planes, planes=planes)
        )

    def forward(self, x):
        out = self.DBB(x)
        return out


class XNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c+l5c+l4c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c+l5c+l4c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_7_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_3_down = self.b1_4_3_down(x1_4_2)
        x1_4_3_same = self.b1_4_3_same(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_up = self.b1_5_2_up(x1_5_1)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)
        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)

        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2(x2_4_1)
        x2_4_3_down = self.b2_4_3_down(x2_4_2)
        x2_4_3_same = self.b2_4_3_same(x2_4_2)

        x2_5_1 = self.b2_4_2_down(x2_4_1)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_up = self.b2_5_2_up(x2_5_1)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_5_2_same, x2_4_3_down), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        x1_4_4 = torch.cat((x1_4_3_same, x2_4_3_same, x2_5_2_up), dim=1)
        x1_4_4 = self.b1_4_4_transition(x1_4_4)
        x1_4_4 = self.b1_4_5(x1_4_4)
        x1_4_4 = torch.cat((x1_4_4, x1_5_3), dim=1)
        x1_4_4 = self.b1_4_6(x1_4_4)
        x1_4_4 = self.b1_4_7_up(x1_4_4)
        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_5_2_same, x1_4_3_down), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        x2_4_4 = torch.cat((x2_4_3_same, x1_4_3_same, x1_5_2_up), dim=1)
        x2_4_4 = self.b2_4_4_transition(x2_4_4)
        x2_4_4 = self.b2_4_5(x2_4_4)
        x2_4_4 = torch.cat((x2_4_4, x2_5_3), dim=1)
        x2_4_4 = self.b2_4_6(x2_4_4)
        x2_4_4 = self.b2_4_7_up(x2_4_4)

        # decode
        # branch1
        x1_3 = torch.cat((x1_3, x1_4_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_3 = torch.cat((x2_3, x2_4_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1


class XNet_1_1_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_1_1_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_3 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_4_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_3 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_4_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4 = self.b1_3_2_down(x1_3)
        x1_4 = self.b1_4_1(x1_4)

        x1_5_1 = self.b1_4_2_down(x1_4)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)
        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)

        x2_4 = self.b2_3_2_down(x2_3)
        x2_4 = self.b2_4_1(x2_4)

        x2_5_1 = self.b2_4_2_down(x2_4)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_5_2_same), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_5_2_same), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        # decode
        # branch1
        x1_4 = torch.cat((x1_4, x1_5_3), dim=1)
        x1_4 = self.b1_4_3(x1_4)
        x1_4 = self.b1_4_4_up(x1_4)

        x1_3 = torch.cat((x1_3, x1_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_4 = torch.cat((x2_4, x2_5_3), dim=1)
        x2_4 = self.b2_4_3(x2_4)
        x2_4 = self.b2_4_4_up(x2_4)

        x2_3 = torch.cat((x2_3, x2_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1

class XNet_1_2_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_1_2_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_3 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_4_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c+l5c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_7_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4 = self.b1_3_2_down(x1_3)
        x1_4 = self.b1_4_1(x1_4)

        x1_5_1 = self.b1_4_2_down(x1_4)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_up = self.b1_5_2_up(x1_5_1)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)
        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)

        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2(x2_4_1)
        x2_4_3_down = self.b2_4_3_down(x2_4_2)
        x2_4_3_same = self.b2_4_3_same(x2_4_2)

        x2_5_1 = self.b2_4_2_down(x2_4_1)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_5_2_same, x2_4_3_down), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_5_2_same), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        x2_4_4 = torch.cat((x2_4_3_same, x1_5_2_up), dim=1)
        x2_4_4 = self.b2_4_4_transition(x2_4_4)
        x2_4_4 = self.b2_4_5(x2_4_4)
        x2_4_4 = torch.cat((x2_4_4, x2_5_3), dim=1)
        x2_4_4 = self.b2_4_6(x2_4_4)
        x2_4_4 = self.b2_4_7_up(x2_4_4)

        # decode
        # branch1
        x1_4 = torch.cat((x1_4, x1_5_3), dim=1)
        x1_4 = self.b1_4_3(x1_4)
        x1_4 = self.b1_4_4_up(x1_4)

        x1_3 = torch.cat((x1_3, x1_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_3 = torch.cat((x2_3, x2_4_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1


class XNet_2_1_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_2_1_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c+l5c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_3 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_4_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_3_down = self.b1_4_3_down(x1_4_2)
        x1_4_3_same = self.b1_4_3_same(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)
        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)

        x2_4 = self.b2_3_2_down(x2_3)
        x2_4 = self.b2_4_1(x2_4)

        x2_5_1 = self.b2_4_2_down(x2_4)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_up = self.b2_5_2_up(x2_5_1)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_5_2_same), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        x1_4_4 = torch.cat((x1_4_3_same, x2_5_2_up), dim=1)
        x1_4_4 = self.b1_4_4_transition(x1_4_4)
        x1_4_4 = self.b1_4_5(x1_4_4)
        x1_4_4 = torch.cat((x1_4_4, x1_5_3), dim=1)
        x1_4_4 = self.b1_4_6(x1_4_4)
        x1_4_4 = self.b1_4_7_up(x1_4_4)
        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_5_2_same, x1_4_3_down), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        # decode
        # branch1
        x1_3 = torch.cat((x1_3, x1_4_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_4 = torch.cat((x2_4, x2_5_3), dim=1)
        x2_4 = self.b2_4_3(x2_4)
        x2_4 = self.b2_4_4_up(x2_4)

        x2_3 = torch.cat((x2_3, x2_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1


class XNet_2_3_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_2_3_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c + l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c + l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_3_up = up_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c+l5c+l4c+l3c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_up_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c+l4c+l3c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_2 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_3_down = down_conv(l3c, l3c)
        self.b2_3_3_down_down = down_conv(l3c, l3c)
        self.b2_3_3_same = same_conv(l3c, l3c)
        self.b2_3_4_transition = transition_conv(l3c+l5c+l4c, l3c)
        self.b2_3_5 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_6 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_7_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c+l5c+l4c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_7_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_3_down = self.b1_4_3_down(x1_4_2)
        x1_4_3_same = self.b1_4_3_same(x1_4_2)
        x1_4_3_up = self.b1_4_3_up(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_up = self.b1_5_2_up(x1_5_1)
        x1_5_2_up_up = self.b1_5_2_up_up(x1_5_2_up)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)

        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3_1 = self.b2_2_2_down(x2_2)
        x2_3_1 = self.b2_3_1(x2_3_1)
        x2_3_2 = self.b2_3_2(x2_3_1)
        x2_3_3_down = self.b2_3_3_down(x2_3_2)
        x2_3_3_down_down = self.b2_3_3_down_down(x2_3_3_down)
        x2_3_3_same = self.b2_3_3_same(x2_3_2)

        x2_4_1 = self.b2_3_2_down(x2_3_1)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2(x2_4_1)
        x2_4_3_down = self.b2_4_3_down(x2_4_2)
        x2_4_3_same = self.b2_4_3_same(x2_4_2)

        x2_5_1 = self.b2_4_2_down(x2_4_1)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_up = self.b2_5_2_up(x2_5_1)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_3_3_down_down, x2_4_3_down, x2_5_2_same), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        x1_4_4 = torch.cat((x1_4_3_same, x2_3_3_down, x2_4_3_same, x2_5_2_up), dim=1)
        x1_4_4 = self.b1_4_4_transition(x1_4_4)
        x1_4_4 = self.b1_4_5(x1_4_4)
        x1_4_4 = torch.cat((x1_4_4, x1_5_3), dim=1)
        x1_4_4 = self.b1_4_6(x1_4_4)
        x1_4_4 = self.b1_4_7_up(x1_4_4)

        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_4_3_down, x1_5_2_same), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        x2_4_4 = torch.cat((x2_4_3_same, x1_4_3_same, x1_5_2_up), dim=1)
        x2_4_4 = self.b2_4_4_transition(x2_4_4)
        x2_4_4 = self.b2_4_5(x2_4_4)
        x2_4_4 = torch.cat((x2_4_4, x2_5_3), dim=1)
        x2_4_4 = self.b2_4_6(x2_4_4)
        x2_4_4 = self.b2_4_7_up(x2_4_4)

        x2_3_4 = torch.cat((x2_3_3_same, x1_4_3_up, x1_5_2_up_up), dim=1)
        x2_3_4 = self.b2_3_4_transition(x2_3_4)
        x2_3_4 = self.b2_3_5(x2_3_4)
        x2_3_4 = torch.cat((x2_3_4, x2_4_4), dim=1)
        x2_3_4 = self.b2_3_6(x2_3_4)
        x2_3_4 = self.b2_3_7_up(x2_3_4)

        # decode
        # branch1
        x1_3 = torch.cat((x1_3, x1_4_4), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_2 = torch.cat((x2_2, x2_3_4), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1


class XNet_3_2_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_3_2_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_2 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_3_down = down_conv(l3c, l3c)
        self.b1_3_3_down_down = down_conv(l3c, l3c)
        self.b1_3_3_same = same_conv(l3c, l3c)
        self.b1_3_4_transition = transition_conv(l3c+l5c+l4c, l3c)
        self.b1_3_5 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_6 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_7_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c+l5c+l4c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_3 = DoubleBasicBlock(l3c + l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c + l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_4_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_3_up = up_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c+l5c+l4c+l3c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_7_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_up_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c+l4c+l3c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3_1 = self.b1_2_2_down(x1_2)
        x1_3_1 = self.b1_3_1(x1_3_1)
        x1_3_2 = self.b1_3_2(x1_3_1)
        x1_3_3_down = self.b1_3_3_down(x1_3_2)
        x1_3_3_down_down = self.b1_3_3_down_down(x1_3_3_down)
        x1_3_3_same = self.b1_3_3_same(x1_3_2)

        x1_4_1 = self.b1_3_2_down(x1_3_1)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_3_down = self.b1_4_3_down(x1_4_2)
        x1_4_3_same = self.b1_4_3_same(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_up = self.b1_5_2_up(x1_5_1)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)

        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3 = self.b2_2_2_down(x2_2)
        x2_3 = self.b2_3_1(x2_3)

        x2_4_1 = self.b2_3_2_down(x2_3)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2(x2_4_1)
        x2_4_3_down = self.b2_4_3_down(x2_4_2)
        x2_4_3_same = self.b2_4_3_same(x2_4_2)
        x2_4_3_up = self.b2_4_3_up(x2_4_2)

        x2_5_1 = self.b2_4_2_down(x2_4_1)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_up = self.b2_5_2_up(x2_5_1)
        x2_5_2_up_up = self.b2_5_2_up_up(x2_5_2_up)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_4_3_down, x2_5_2_same), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        x1_4_4 = torch.cat((x1_4_3_same, x2_4_3_same, x2_5_2_up), dim=1)
        x1_4_4 = self.b1_4_4_transition(x1_4_4)
        x1_4_4 = self.b1_4_5(x1_4_4)
        x1_4_4 = torch.cat((x1_4_4, x1_5_3), dim=1)
        x1_4_4 = self.b1_4_6(x1_4_4)
        x1_4_4 = self.b1_4_7_up(x1_4_4)

        x1_3_4 = torch.cat((x1_3_3_same, x2_4_3_up, x2_5_2_up_up), dim=1)
        x1_3_4 = self.b1_3_4_transition(x1_3_4)
        x1_3_4 = self.b1_3_5(x1_3_4)
        x1_3_4 = torch.cat((x1_3_4, x1_4_4), dim=1)
        x1_3_4 = self.b1_3_6(x1_3_4)
        x1_3_4 = self.b1_3_7_up(x1_3_4)

        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_3_3_down_down, x1_4_3_down, x1_5_2_same), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        x2_4_4 = torch.cat((x2_4_3_same, x1_3_3_down, x1_4_3_same, x1_5_2_up), dim=1)
        x2_4_4 = self.b2_4_4_transition(x2_4_4)
        x2_4_4 = self.b2_4_5(x2_4_4)
        x2_4_4 = torch.cat((x2_4_4, x2_5_3), dim=1)
        x2_4_4 = self.b2_4_6(x2_4_4)
        x2_4_4 = self.b2_4_7_up(x2_4_4)

        # decode
        # branch1
        x1_2 = torch.cat((x1_2, x1_3_4), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_3 = torch.cat((x2_3, x2_4_4), dim=1)
        x2_3 = self.b2_3_3(x2_3)
        x2_3 = self.b2_3_4_up(x2_3)

        x2_2 = torch.cat((x2_2, x2_3), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1



class XNet_3_3_m(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_3_3_m, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_2 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_3_down = down_conv(l3c, l3c)
        self.b1_3_3_down_down = down_conv(l3c, l3c)
        self.b1_3_3_same = same_conv(l3c, l3c)
        self.b1_3_4_transition = transition_conv(l3c+l5c+l4c+l3c, l3c)
        self.b1_3_5 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_6 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_7_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_3_down = down_conv(l4c, l4c)
        self.b1_4_3_same = same_conv(l4c, l4c)
        self.b1_4_3_up = up_conv(l4c, l4c)
        self.b1_4_4_transition = transition_conv(l4c+l5c+l4c+l3c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_2_up = up_conv(l5c, l5c)
        self.b1_5_2_up_up = up_conv(l5c, l5c)
        self.b1_5_2_same = same_conv(l5c, l5c)
        self.b1_5_3_transition = transition_conv(l5c+l5c+l4c+l3c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # branch2
        # branch2_layer1
        self.b2_1_1 = nn.Sequential(
            conv3x3(1, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b2_1_2_down = down_conv(l1c, l2c)
        self.b2_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b2_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch2_layer2
        self.b2_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b2_2_2_down = down_conv(l2c, l3c)
        self.b2_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b2_2_4_up = up_conv(l2c, l1c)
        # branch2_layer3
        self.b2_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_2_down = down_conv(l3c, l4c)
        self.b2_3_2 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_3_down = down_conv(l3c, l3c)
        self.b2_3_3_down_down = down_conv(l3c, l3c)
        self.b2_3_3_same = same_conv(l3c, l3c)
        self.b2_3_4_transition = transition_conv(l3c+l5c+l4c+l3c, l3c)
        self.b2_3_5 = DoubleBasicBlock(l3c, l3c)
        self.b2_3_6 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b2_3_7_up = up_conv(l3c, l2c)
        # branch2_layer4
        self.b2_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_2_down = down_conv(l4c, l5c)
        self.b2_4_2 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_3_down = down_conv(l4c, l4c)
        self.b2_4_3_same = same_conv(l4c, l4c)
        self.b2_4_3_up = up_conv(l4c, l4c)
        self.b2_4_4_transition = transition_conv(l4c+l5c+l4c+l3c, l4c)
        self.b2_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b2_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b2_4_7_up = up_conv(l4c, l3c)
        # branch2_layer5
        self.b2_5_1 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_2_up = up_conv(l5c, l5c)
        self.b2_5_2_up_up = up_conv(l5c, l5c)
        self.b2_5_2_same = same_conv(l5c, l5c)
        self.b2_5_3_transition = transition_conv(l5c+l5c+l4c+l3c, l5c)
        self.b2_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b2_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1, input2):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3_1 = self.b1_2_2_down(x1_2)
        x1_3_1 = self.b1_3_1(x1_3_1)
        x1_3_2 = self.b1_3_2(x1_3_1)
        x1_3_3_down = self.b1_3_3_down(x1_3_2)
        x1_3_3_down_down = self.b1_3_3_down_down(x1_3_3_down)
        x1_3_3_same = self.b1_3_3_same(x1_3_2)

        x1_4_1 = self.b1_3_2_down(x1_3_1)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_3_down = self.b1_4_3_down(x1_4_2)
        x1_4_3_same = self.b1_4_3_same(x1_4_2)
        x1_4_3_up = self.b1_4_3_up(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_2_up = self.b1_5_2_up(x1_5_1)
        x1_5_2_up_up = self.b1_5_2_up_up(x1_5_2_up)
        x1_5_2_same = self.b1_5_2_same(x1_5_1)

        # branch2
        x2_1 = self.b2_1_1(input2)

        x2_2 = self.b2_1_2_down(x2_1)
        x2_2 = self.b2_2_1(x2_2)

        x2_3_1 = self.b2_2_2_down(x2_2)
        x2_3_1 = self.b2_3_1(x2_3_1)
        x2_3_2 = self.b2_3_2(x2_3_1)
        x2_3_3_down = self.b2_3_3_down(x2_3_2)
        x2_3_3_down_down = self.b2_3_3_down_down(x2_3_3_down)
        x2_3_3_same = self.b2_3_3_same(x2_3_2)

        x2_4_1 = self.b2_3_2_down(x2_3_1)
        x2_4_1 = self.b2_4_1(x2_4_1)
        x2_4_2 = self.b2_4_2(x2_4_1)
        x2_4_3_down = self.b2_4_3_down(x2_4_2)
        x2_4_3_same = self.b2_4_3_same(x2_4_2)
        x2_4_3_up = self.b2_4_3_up(x2_4_2)

        x2_5_1 = self.b2_4_2_down(x2_4_1)
        x2_5_1 = self.b2_5_1(x2_5_1)
        x2_5_2_up = self.b2_5_2_up(x2_5_1)
        x2_5_2_up_up = self.b2_5_2_up_up(x2_5_2_up)
        x2_5_2_same = self.b2_5_2_same(x2_5_1)

        # merge
        # branch1
        x1_5_3 = torch.cat((x1_5_2_same, x2_3_3_down_down, x2_4_3_down, x2_5_2_same), dim=1)
        x1_5_3 = self.b1_5_3_transition(x1_5_3)
        x1_5_3 = self.b1_5_4(x1_5_3)
        x1_5_3 = self.b1_5_5_up(x1_5_3)

        x1_4_4 = torch.cat((x1_4_3_same, x2_3_3_down, x2_4_3_same, x2_5_2_up), dim=1)
        x1_4_4 = self.b1_4_4_transition(x1_4_4)
        x1_4_4 = self.b1_4_5(x1_4_4)
        x1_4_4 = torch.cat((x1_4_4, x1_5_3), dim=1)
        x1_4_4 = self.b1_4_6(x1_4_4)
        x1_4_4 = self.b1_4_7_up(x1_4_4)

        x1_3_4 = torch.cat((x1_3_3_same, x2_3_3_same, x2_4_3_up, x2_5_2_up_up), dim=1)
        x1_3_4 = self.b1_3_4_transition(x1_3_4)
        x1_3_4 = self.b1_3_5(x1_3_4)
        x1_3_4 = torch.cat((x1_3_4, x1_4_4), dim=1)
        x1_3_4 = self.b1_3_6(x1_3_4)
        x1_3_4 = self.b1_3_7_up(x1_3_4)

        # branch2
        x2_5_3 = torch.cat((x2_5_2_same, x1_3_3_down_down, x1_4_3_down, x1_5_2_same), dim=1)
        x2_5_3 = self.b2_5_3_transition(x2_5_3)
        x2_5_3 = self.b2_5_4(x2_5_3)
        x2_5_3 = self.b2_5_5_up(x2_5_3)

        x2_4_4 = torch.cat((x2_4_3_same, x1_3_3_down, x1_4_3_same, x1_5_2_up), dim=1)
        x2_4_4 = self.b2_4_4_transition(x2_4_4)
        x2_4_4 = self.b2_4_5(x2_4_4)
        x2_4_4 = torch.cat((x2_4_4, x2_5_3), dim=1)
        x2_4_4 = self.b2_4_6(x2_4_4)
        x2_4_4 = self.b2_4_7_up(x2_4_4)

        x2_3_4 = torch.cat((x2_3_3_same, x1_3_3_same, x1_4_3_up, x1_5_2_up_up), dim=1)
        x2_3_4 = self.b2_3_4_transition(x2_3_4)
        x2_3_4 = self.b2_3_5(x2_3_4)
        x2_3_4 = torch.cat((x2_3_4, x2_4_4), dim=1)
        x2_3_4 = self.b2_3_6(x2_3_4)
        x2_3_4 = self.b2_3_7_up(x2_3_4)

        # decode
        # branch1
        x1_2 = torch.cat((x1_2, x1_3_4), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)
        # branch2
        x2_2 = torch.cat((x2_2, x2_3_4), dim=1)
        x2_2 = self.b2_2_3(x2_2)
        x2_2 = self.b2_2_4_up(x2_2)

        x2_1 = torch.cat((x2_1, x2_2), dim=1)
        x2_1 = self.b2_1_3(x2_1)
        x2_1 = self.b2_1_4(x2_1)

        return x1_1, x2_1

class XNet_sb(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XNet_sb, self).__init__()

        l1c, l2c, l3c, l4c, l5c = 64, 128, 256, 512, 1024

        # branch1
        # branch1_layer1
        self.b1_1_1 = nn.Sequential(
            conv3x3(in_channels, l1c),
            conv3x3(l1c, l1c),
            BasicBlock(l1c, l1c)
        )
        self.b1_1_2_down = down_conv(l1c, l2c)
        self.b1_1_3 = DoubleBasicBlock(l1c+l1c, l1c, nn.Sequential(conv1x1(in_planes=l1c+l1c, out_planes=l1c), BatchNorm2d(l1c, momentum=BN_MOMENTUM)))
        self.b1_1_4 = nn.Conv2d(l1c, num_classes, kernel_size=1, stride=1, padding=0)
        # branch1_layer2
        self.b1_2_1 = DoubleBasicBlock(l2c, l2c)
        self.b1_2_2_down = down_conv(l2c, l3c)
        self.b1_2_3 = DoubleBasicBlock(l2c+l2c, l2c, nn.Sequential(conv1x1(in_planes=l2c+l2c, out_planes=l2c), BatchNorm2d(l2c, momentum=BN_MOMENTUM)))
        self.b1_2_4_up = up_conv(l2c, l1c)
        # branch1_layer3
        self.b1_3_1 = DoubleBasicBlock(l3c, l3c)
        self.b1_3_2_down = down_conv(l3c, l4c)
        self.b1_3_3 = DoubleBasicBlock(l3c+l3c, l3c, nn.Sequential(conv1x1(in_planes=l3c+l3c, out_planes=l3c), BatchNorm2d(l3c, momentum=BN_MOMENTUM)))
        self.b1_3_4_up = up_conv(l3c, l2c)
        # branch1_layer4
        self.b1_4_1 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_2_down = down_conv(l4c, l5c)
        self.b1_4_2 = DoubleBasicBlock(l4c, l4c)
        # self.b1_4_3_down = down_conv(l4c, l4c)
        # self.b1_4_3_same = same_conv(l4c, l4c)
        # self.b1_4_4_transition = transition_conv(l4c, l4c)
        self.b1_4_5 = DoubleBasicBlock(l4c, l4c)
        self.b1_4_6 = DoubleBasicBlock(l4c+l4c, l4c, nn.Sequential(conv1x1(in_planes=l4c+l4c, out_planes=l4c), BatchNorm2d(l4c, momentum=BN_MOMENTUM)))
        self.b1_4_7_up = up_conv(l4c, l3c)
        # branch1_layer5
        self.b1_5_1 = DoubleBasicBlock(l5c, l5c)
        # self.b1_5_2_up = up_conv(l5c, l5c)
        # self.b1_5_2_same = same_conv(l5c, l5c)
        # self.b1_5_3_transition = transition_conv(l5c+l5c+l4c, l5c)
        self.b1_5_4 = DoubleBasicBlock(l5c, l5c)
        self.b1_5_5_up = up_conv(l5c, l4c)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, InPlaceABN):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input1):
        # code
        # branch1
        x1_1 = self.b1_1_1(input1)

        x1_2 = self.b1_1_2_down(x1_1)
        x1_2 = self.b1_2_1(x1_2)

        x1_3 = self.b1_2_2_down(x1_2)
        x1_3 = self.b1_3_1(x1_3)

        x1_4_1 = self.b1_3_2_down(x1_3)
        x1_4_1 = self.b1_4_1(x1_4_1)
        x1_4_2 = self.b1_4_2(x1_4_1)
        x1_4_2 = self.b1_4_5(x1_4_2)
        # x1_4_3_down = self.b1_4_3_down(x1_4_2)
        # x1_4_3_same = self.b1_4_3_same(x1_4_2)

        x1_5_1 = self.b1_4_2_down(x1_4_1)
        x1_5_1 = self.b1_5_1(x1_5_1)
        x1_5_1 = self.b1_5_4(x1_5_1)
        x1_5_1 = self.b1_5_5_up(x1_5_1)

        # x1_5_2_up = self.b1_5_2_up(x1_5_1)
        # x1_5_2_same = self.b1_5_2_same(x1_5_1)

        # decode
        # branch1
        x1_4_2 = torch.cat((x1_4_2, x1_5_1), dim=1)
        x1_4_2 = self.b1_4_6(x1_4_2)
        x1_4_2 = self.b1_4_7_up(x1_4_2)

        x1_3 = torch.cat((x1_3, x1_4_2), dim=1)
        x1_3 = self.b1_3_3(x1_3)
        x1_3 = self.b1_3_4_up(x1_3)

        x1_2 = torch.cat((x1_2, x1_3), dim=1)
        x1_2 = self.b1_2_3(x1_2)
        x1_2 = self.b1_2_4_up(x1_2)

        x1_1 = torch.cat((x1_1, x1_2), dim=1)
        x1_1 = self.b1_1_3(x1_1)
        x1_1 = self.b1_1_4(x1_1)

        return x1_1


# if __name__ == '__main__':
#     model = XNet(1, 10)
    # total = sum([param.nelement() for param in model.parameters()])
    # from thop import profile, clever_format
    #
    # input = torch.randn(1, 1, 128, 128)
    # flops, params = profile(model, inputs=(input, input, ))
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs)
    # print(params)
    # print(total)
    # model.eval()
    # input1 = torch.rand(2,3,256,256)
    # input2 = torch.rand(2,1,256,256)
    # x1_1, x2_1 = model(input1, input2)
    # output1 = x1_1.data.cpu().numpy()
    # output2 = x2_1.data.cpu().numpy()
    # # print(output)
    # print(output1.shape)
    # print(output2.shape)
