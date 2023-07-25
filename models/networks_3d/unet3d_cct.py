import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.distributions.uniform import Uniform

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

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class Decoder(nn.Module):
    def __init__(self, features, out_channels):
        super(Decoder, self).__init__()

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Decoder._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Decoder._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Decoder._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Decoder._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x5, x4, x3, x2, x1):

        dec4 = self.upconv4(x5)
        dec4 = torch.cat((dec4, x4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, x3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, x2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, x1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)

        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class UNet3D_CCT(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D_CCT, self).__init__()

        features = init_features
        self.encoder1 = UNet3D_CCT._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D_CCT._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D_CCT._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D_CCT._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D_CCT._block(features * 8, features * 16, name="bottleneck")

        self.main_decoder = Decoder(features, out_channels)

        self.aux_decoder1 = Decoder(features, out_channels)
        self.aux_decoder2 = Decoder(features, out_channels)
        self.aux_decoder3 = Decoder(features, out_channels)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        main_seg = self.main_decoder(bottleneck, enc4, enc3, enc2, enc1)

        aux_seg1 = self.main_decoder(FeatureNoise()(bottleneck), FeatureNoise()(enc4), FeatureNoise()(enc3), FeatureNoise()(enc2), FeatureNoise()(enc1))
        aux_seg2 = self.main_decoder(Dropout(bottleneck), Dropout(enc4), Dropout(enc3), Dropout(enc2), Dropout(enc1))
        aux_seg3 = self.main_decoder(FeatureDropout(bottleneck), FeatureDropout(enc4), FeatureDropout(enc3), FeatureDropout(enc2), FeatureDropout(enc1))

        return main_seg, aux_seg1, aux_seg2, aux_seg3

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class UNet3D_CCT_min(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D_CCT_min, self).__init__()

        features = init_features
        self.encoder1 = UNet3D_CCT._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D_CCT._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D_CCT._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D_CCT._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D_CCT._block(features * 8, features * 16, name="bottleneck")

        self.main_decoder = Decoder(features, out_channels)

        self.aux_decoder1 = Decoder(features, out_channels)
        self.aux_decoder2 = Decoder(features, out_channels)
        self.aux_decoder3 = Decoder(features, out_channels)


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        main_seg = self.main_decoder(bottleneck, enc4, enc3, enc2, enc1)

        aux_seg1 = self.main_decoder(FeatureNoise()(bottleneck), FeatureNoise()(enc4), FeatureNoise()(enc3), FeatureNoise()(enc2), FeatureNoise()(enc1))
        aux_seg2 = self.main_decoder(Dropout(bottleneck), Dropout(enc4), Dropout(enc3), Dropout(enc2), Dropout(enc1))
        aux_seg3 = self.main_decoder(FeatureDropout(bottleneck), FeatureDropout(enc4), FeatureDropout(enc3), FeatureDropout(enc2), FeatureDropout(enc1))

        return main_seg, aux_seg1, aux_seg2, aux_seg3

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def unet3d_cct(in_channels, num_classes):
    model = UNet3D_CCT(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def unet3d_cct_min(in_channels, num_classes):
    model = UNet3D_CCT_min(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

# if __name__ == '__main__':
#     model = unet3d_cct(1,10)
#     model.eval()
#     input = torch.rand(2, 1, 128, 128, 128)
#     output, aux_output1, aux_output2, aux_output3 = model(input)
#     output = output.data.cpu().numpy()
#     # print(output)
#     print(output.shape)
