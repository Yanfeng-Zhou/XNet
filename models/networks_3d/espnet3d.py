import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from loss.loss_function import segmentation_loss
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        k = 4
        n = int(nOut/k)
        n1 = nOut - (k-1)*n
        self.c1 = nn.Sequential(CBR(nIn, n, 1, 1), C(n, n, 3, 2))
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 3)
        self.d8 = CDilated(n, n, 3, 1, 4)
        self.bn = BR(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3],1)
        if input.size() == combine.size():
            combine = input + combine
        output = self.bn(combine)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv3d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d, groups=groups)
        #self.bn = nn.BatchNorm3d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        return self.conv(input)
        #return self.bn(output)


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool3d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        k = 4
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = CBR(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, stride, 1)
        self.d2 = CDilated(n, n, 3, stride, 1)
        self.d4 = CDilated(n, n, 3, stride, 2)
        self.d8 = CDilated(n, n, 3, stride, 2)
        self.bn = nn.BatchNorm3d(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = self.bn(torch.cat([d1, add1, add2, add3], 1))
        if input.size() == combine.size():
            combine = input + combine
        output = F.relu(combine, inplace=True)
        return output

class ASPBlock(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        self.d1 = CB(nIn, nOut, 3, 1)
        self.d2 = CB(nIn, nOut, 5, 1)
        self.d4 = CB(nIn, nOut, 7, 1)
        self.d8 = CB(nIn, nOut, 9, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        d1 = self.d1(input)
        d2 = self.d2(input)
        d3 = self.d4(input)
        d4 = self.d8(input)

        combine = d1 + d2 + d3 + d4
        if input.size() == combine.size():
            combine = input + combine
        output = self.act(combine)
        return output


class UpSampler(nn.Module):
    '''
    Up-sample the feature maps by 2
    '''
    def __init__(self, nIn, nOut):
        super().__init__()
        self.up = CBR(nIn, nOut, 3, 1)

    def forward(self, inp):
        return F.upsample(self.up(inp), mode='trilinear', scale_factor=2, align_corners=True)


class PSPDec(nn.Module):
    '''
    Inspired or Adapted from Pyramid Scene Network paper
    '''

    def __init__(self, nIn, nOut, downSize):
        super().__init__()
        self.scale = downSize
        self.features = CBR(nIn, nOut, 3, 1)
    def forward(self, x):
        assert x.dim() == 5
        inp_size = x.size()
        out_dim1, out_dim2, out_dim3 = int(inp_size[2] * self.scale), int(inp_size[3] * self.scale), int(inp_size[4] * self.scale)
        x_down = F.adaptive_avg_pool3d(x, output_size=(out_dim1, out_dim2, out_dim3))
        return F.upsample(self.features(x_down), size=(inp_size[2], inp_size[3], inp_size[4]), mode='trilinear', align_corners=True)

class ESPNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.input1 = InputProjectionA(1)
        self.input2 = InputProjectionA(1)

        initial = 16 # feature maps at level 1
        config = [32, 128, 256, 256] # feature maps at level 2 and onwards
        reps = [2, 2, 3]

        ### ENCODER

        # all dimensions are listed with respect to an input  of size 4 x 128 x 128 x 128
        self.level0 = CBR(in_channels, initial, 7, 2) # initial x 64 x 64 x64
        self.level1 = nn.ModuleList()
        for i in range(reps[0]):
            if i==0:
                self.level1.append(DilatedParllelResidualBlockB1(initial, config[0]))  # config[0] x 64 x 64 x64
            else:
                self.level1.append(DilatedParllelResidualBlockB1(config[0], config[0]))  # config[0] x 64 x 64 x64

        # downsample the feature maps
        self.level2 = DilatedParllelResidualBlockB1(config[0], config[1], stride=2) # config[1] x 32 x 32 x 32
        self.level_2 = nn.ModuleList()
        for i in range(0, reps[1]):
            self.level_2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        # downsample the feature maps
        self.level3_0 = DilatedParllelResidualBlockB1(config[1], config[2], stride=2) # config[2] x 16 x 16 x 16
        self.level_3 = nn.ModuleList()
        for i in range(0, reps[2]):
            self.level_3.append(DilatedParllelResidualBlockB1(config[2], config[2])) # config[2] x 16 x 16 x 16


        ### DECODER

        # upsample the feature maps
        self.up_l3_l2 = UpSampler(config[2], config[1])  # config[1] x 32 x 32 x 32
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l2 = DilatedParllelResidualBlockB1(2 * config[1], config[1]) # config[1] x 32 x 32 x 32
        self.dec_l2 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        self.up_l2_l1 = UpSampler(config[1], config[0])  # config[0] x 64 x 64 x 64
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l1 = DilatedParllelResidualBlockB1(2*config[0], config[0]) # config[0] x 64 x 64 x 64
        self.dec_l1 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l1.append(DilatedParllelResidualBlockB1(config[0], config[0])) # config[0] x 64 x 64 x 64

        self.dec_l1.append(CBR(config[0], num_classes, 3, 1)) # classes x 64 x 64 x 64
        # We use ESP block without reduction step because the number  of input feature maps are very small (i.e. 4 in
        # our case)
        self.dec_l1.append(ASPBlock(num_classes, num_classes))

        # Using PSP module to learn the representations at different scales
        self.pspModules = nn.ModuleList()
        scales = [0.2, 0.4, 0.6, 0.8]
        for sc in scales:
             self.pspModules.append(PSPDec(num_classes, num_classes, sc))

        # Classifier
        self.classifier = self.classifier = nn.Sequential(
             CBR((len(scales) + 1) * num_classes, num_classes, 3, 1),
             ASPBlock(num_classes, num_classes), # classes x 64 x 64 x 64
             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), # classes x 128 x 128 x 128
             CBR(num_classes, num_classes, 7, 1), # classes x 128 x 128 x 128
             C(num_classes, num_classes, 1, 1) # classes x 128 x 128 x 128
        )
        #

        for m in self.modules():
             if isinstance(m, nn.Conv3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             if isinstance(m, nn.ConvTranspose3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             elif isinstance(m, nn.BatchNorm3d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()

    def forward(self, input1, inp_res=(128, 128, 128), inpSt2=False):
        dim0 = input1.size(2)
        dim1 = input1.size(3)
        dim2 = input1.size(4)

        if self.training or inp_res is None:
            # input resolution should be divisible by 8
            inp_res = (math.ceil(dim0 / 8) * 8, math.ceil(dim1 / 8) * 8,
                       math.ceil(dim2 / 8) * 8)
        if inp_res:
            input1 = F.adaptive_avg_pool3d(input1, output_size=inp_res)

        out_l0 = self.level0(input1)

        for i, layer in enumerate(self.level1): #64
            if i == 0:
                out_l1 = layer(out_l0)
            else:
                out_l1 = layer(out_l1)

        out_l2_down = self.level2(out_l1) #32
        for i, layer in enumerate(self.level_2):
            if i == 0:
                out_l2 = layer(out_l2_down)
            else:
                out_l2 = layer(out_l2)
        del out_l2_down

        out_l3_down = self.level3_0(out_l2) #16
        for i, layer in enumerate(self.level_3):
            if i == 0:
                out_l3 = layer(out_l3_down)
            else:
                out_l3 = layer(out_l3)
        del out_l3_down

        dec_l3_l2 = self.up_l3_l2(out_l3)
        merge_l2 = self.merge_l2(torch.cat([dec_l3_l2, out_l2], 1))
        for i, layer in enumerate(self.dec_l2):
            if i == 0:
                dec_l2 = layer(merge_l2)
            else:
                dec_l2 = layer(dec_l2)

        dec_l2_l1 = self.up_l2_l1(dec_l2)
        merge_l1 = self.merge_l1(torch.cat([dec_l2_l1, out_l1], 1))
        for i, layer in enumerate(self.dec_l1):
            if i == 0:
                dec_l1 = layer(merge_l1)
            else:
                dec_l1 = layer(dec_l1)

        psp_outs = dec_l1.clone()
        for layer in self.pspModules:
            out_psp = layer(dec_l1)
            psp_outs = torch.cat([psp_outs, out_psp], 1)

        decoded = self.classifier(psp_outs)
        return F.upsample(decoded, size=(dim0, dim1, dim2), mode='trilinear', align_corners=True)

def espnet3d(in_channels, num_classes):
    model = ESPNet(in_channels, num_classes)
    return model


# if __name__ == '__main__':
#
#     criterion = segmentation_loss('dice', False)
#
#     mask = torch.ones(2, 96, 48, 96).long()
#     model = espnet3d(1, 10)
#     model.train()
#     input = torch.rand(2, 1, 96, 48, 96)
#     output = model(input)
#     loss_train = criterion(output, mask)
#     output = output.data.cpu().numpy()
#     loss_train.backward()
#     print(output.shape)
#     print(loss_train)