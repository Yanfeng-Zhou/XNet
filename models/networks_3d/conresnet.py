import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from torch.nn import init
# from loss.loss_function import segmentation_loss

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), dilation=(1,1,1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class ConResAtt(nn.Module):
    def __init__(self, in_channels, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(ConResAtt, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer

        self.relu = nn.ReLU(inplace=True)

        self.gn_seg = nn.GroupNorm(8, in_planes)
        self.conv_seg = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res = nn.GroupNorm(8, out_planes)
        self.conv_res = conv3x3x3(out_planes, out_planes, kernel_size=(1,1,1),
                               stride=(1, 1, 1), padding=(0,0,0),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.conv_res1 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.conv_res2 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_planes)
        self.conv_mp_first = conv3x3x3(in_channels, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                              stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                              dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.conv_mp = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

    def _res(self, x):  # bs, channel, D, W, H

        bs, channel, depth, heigt, width = x.shape
        # x_copy = torch.zeros_like(x).cuda()
        x_copy = torch.zeros_like(x)
        x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
        res = x - x_copy
        res[:, :, 0, :, :] = 0
        res = torch.abs(res)
        return res

    def forward(self, input):
        x1, x2 = input
        if self.first_layer:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)

            x2 = self.conv_mp_first(x2)
            x2 = x2 + res

        else:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)


            if self.in_planes != self.out_planes:
                x2 = self.gn_mp(x2)
                x2 = self.relu(x2)
                x2 = self.conv_mp(x2)

            x2 = x2 + res

        x2 = self.gn_res1(x2)
        x2 = self.relu(x2)
        x2 = self.conv_res1(x2)

        x1 = x1*(1 + torch.sigmoid(x2))

        return [x1, x2]


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.ReLU(inplace=True)

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class ConResNet(nn.Module):
    def __init__(self, in_channels, num_classes, shape, block, layers, weight_std=False):
        self.shape = shape
        self.weight_std = weight_std
        super(ConResNet, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(in_channels, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2,2,2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )

        self.seg_x4 = nn.Sequential(
            ConResAtt(in_channels, 128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            ConResAtt(in_channels, 64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            ConResAtt(in_channels, 32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, num_classes, kernel_size=1)
        )

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3x3(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))
        return nn.Sequential(*layers)


    def forward(self, x, x_res):

        ## encoder
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)

        x = self.layer4(x)

        x = self.fusionConv(x)

        ## decoder
        res_x4 = F.interpolate(x_res, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = seg_x4 + skip3
        seg_x4, res_x4 = self.seg_x4([seg_x4, res_x4])

        res_x2 = F.interpolate(res_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = F.interpolate(seg_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = seg_x2 + skip2
        seg_x2, res_x2 = self.seg_x2([seg_x2, res_x2])

        res_x1 = F.interpolate(res_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = F.interpolate(seg_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = seg_x1 + skip1
        seg_x1, res_x1 = self.seg_x1([seg_x1, res_x1])

        seg = self.seg_cls(seg_x1)
        res = self.res_cls(res_x1)
        resx2 = self.resx2_cls(res_x2)
        resx4 = self.resx4_cls(res_x4)

        resx2 = F.interpolate(resx2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        resx4 = F.interpolate(resx4, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)

        return [seg, res, resx2, resx4]


def conresnet(in_channels, num_classes, **kwargs):
    model = ConResNet(in_channels, num_classes, kwargs['img_shape'], NoBottleneck, [1, 2, 2, 2, 2])
    init_weights(model, 'kaiming')
    return model

# if __name__ == '__main__':
#
#     criterion = segmentation_loss('dice', False)
#     mask = torch.ones(5, 64, 64, 64).long()
#     model = conresnet(1, 10, img_shape=(64, 64, 64))
#     model.train()
#     output = model(torch.rand(5, 1, 64, 64, 64), torch.rand(5, 1, 64, 64, 64))
#
#     loss_train_1 = criterion(output[0], mask)
#     loss_train_2 = criterion(output[1], mask)
#     loss_train_3 = criterion(output[2], mask)
#     loss_train_4 = criterion(output[3], mask)
#
#     loss_train_1.backward()
#
#     print(output[0].data.cpu().numpy().shape)
#     print(output[1].data.cpu().numpy().shape)
#     print(output[2].data.cpu().numpy().shape)
#     print(output[3].data.cpu().numpy().shape)
#     print(loss_train_1)
#     print(loss_train_2)
#     print(loss_train_3)
#     print(loss_train_4)


