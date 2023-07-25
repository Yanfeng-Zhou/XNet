import numpy as np
import math, pywt
import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function
from collections import OrderedDict
from itertools import islice
import operator

class My_DownSampling_SC(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = 2, padding = (0,0)):
        super(My_DownSampling_SC, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, input):
        return self.conv(input), input


class My_DownSampling_MP(nn.Module):
    def __init__(self, stride = 2, kernel_size = 2):
        super(My_DownSampling_MP, self).__init__()
        self.maxp = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, return_indices = False)

    def forward(self, input):
        return self.maxp(input), input


class My_UpSampling_SC(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = 2, padding = (0,0)):
        super(My_UpSampling_SC, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, input, feature_map):
        return torch.cat((self.conv(input), feature_map), dim = 1)


class My_DownSampling_DWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(My_DownSampling_DWT, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return LL, LH, HL, HH, input


class My_UpSampling_IDWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(My_UpSampling_IDWT, self).__init__()
        self.idwt = IDWT_2D(wavename = wavename)

    def forward(self, LL, LH, HL, HH, feature_map):
        return torch.cat((self.idwt(LL, LH, HL, HH), feature_map), dim = 1)



class My_Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        self.output = []
        for module in self._modules.values():
            input = module(input)
            if isinstance(input, tuple):
                assert len(input) == 4 or len(input) == 2 or len(input) == 5
                self.output.append(input[1:])
                input = input[0]
        if self.output != []:
            return input, self.output
        else:
            return input


class My_Sequential_re(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential_re, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.output = []

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential_re, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, *input):
        LL = input[0]
        index = 1
        for module in self._modules.values():
            if isinstance(module, My_UpSampling_IDWT):
                LH = input[index]
                HL = input[index + 1]
                HH = input[index + 2]
                feature_map = input[index + 3]
                LL = module(LL, LH, HL, HH, feature_map = feature_map)
                index += 4
            elif isinstance(module, IDWT_2D) or 'idwt' in dir(module):
                LH = input[index]
                HL = input[index + 1]
                HH = input[index + 2]
                LL = module(LL, LH, HL, HH)
                index += 3
            elif isinstance(module, nn.MaxUnpool2d):
                indices = input[index]
                LL = module(input = LL, indices = indices)
                #_, _, h, w = LL.size()
                #LL = F.interpolate(LL, size = (2*h, 2*w), mode = 'bilinear', align_corners = True)
                index += 1
            elif isinstance(module, My_UpSampling_SC):
                feature_map = input[index]
                LL = module(input = LL, feature_map = feature_map)
                index += 1
            else:
                LL = module(LL)
        return LL


class DWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        L = torch.matmul(input, matrix_Low.t())
        H = torch.matmul(input, matrix_High.t())
        return L, H
    @staticmethod
    def backward(ctx, grad_L, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.add(torch.matmul(grad_L, matrix_L), torch.matmul(grad_H, matrix_H))
        return grad_input, None, None


class IDWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input_L, input_H, matrix_L, matrix_H):
        ctx.save_for_backward(matrix_L, matrix_H)
        output = torch.add(torch.matmul(input_L, matrix_L), torch.matmul(input_H, matrix_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_L, matrix_H = ctx.saved_variables
        grad_L = torch.matmul(grad_output, matrix_L.t())
        grad_H = torch.matmul(grad_output, matrix_H.t())
        return grad_L, grad_H, None, None


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class DWTFunction_2D_tiny(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        return LL
    @staticmethod
    def backward(ctx, grad_LL):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
        grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
        return grad_input, None, None, None, None


class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L), torch.matmul(matrix_High_0.t(), H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_tensors
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None


class DWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0 = 2, dim1 = 3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0 = 2, dim1 = 3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0 = 2, dim1 = 3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0 = 2, dim1 = 3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0 = 2, dim1 = 3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0 = 2, dim1 = 3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0 = 2, dim1 = 3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0 = 2, dim1 = 3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                      grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None


class IDWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input_LLL, input_LLH, input_LHL, input_LHH,
                     input_HLL, input_HLH, input_HHL, input_HHH,
                     matrix_Low_0, matrix_Low_1, matrix_Low_2,
                     matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        input_LL = torch.add(torch.matmul(matrix_Low_2.t(), input_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_LH = torch.add(torch.matmul(matrix_Low_2.t(), input_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HL = torch.add(torch.matmul(matrix_Low_2.t(), input_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HH = torch.add(torch.matmul(matrix_Low_2.t(), input_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        input_H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), input_L), torch.matmul(matrix_High_0.t(), input_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.matmul(grad_L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.matmul(grad_H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.matmul(grad_H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_LLL = torch.matmul(matrix_Low_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_LLH = torch.matmul(matrix_Low_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_LHL = torch.matmul(matrix_Low_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_LHH = torch.matmul(matrix_Low_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        grad_HLL = torch.matmul(matrix_High_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_HLH = torch.matmul(matrix_High_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_HHL = torch.matmul(matrix_High_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_HHH = torch.matmul(matrix_High_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        return grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH, None, None, None, None, None, None


class DWT_1D(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, input):
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        #assert self.input_height > self.band_length
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  L -- (N, C, L/2)
            H -- (N, C, L/2)
    output: (N, C, L)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重建所需低频滤波器组
        :param band_high: 小波重建所需高频滤波器组
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        #assert self.input_height > self.band_length
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D(Module):
    """
    input: (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
              LH: (N, C, H/2, W/2)
              HL: (N, C, H/2, W/2)
              HH: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_2D_tiny(Module):
    """
    input: (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
    """
    input -- LL: (N, C, H/2, W/2)
             LH: (N, C, H/2, W/2)
             HL: (N, C, H/2, W/2)
             HH: (N, C, H/2, W/2)
    output: (N, C, H, W)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重建所需低频滤波器组
        :param band_high: 小波重建所需高频滤波器组
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_3D(Module):
    """
    input: (N, C, D, H, W)
    output: -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_low_2 = torch.tensor(matrix_h_2)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)
            self.matrix_high_2 = torch.tensor(matrix_g_2)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


class IDWT_3D(Module):
    """
    input:  -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    output: (N, C, D, H, W)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重构所用低频滤波器组
        :param band_high: 小波重构所用高频滤波器组
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_low_2 = torch.tensor(matrix_h_2)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)
            self.matrix_high_2 = torch.tensor(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width = LLL.size()[-1] + HHH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


# if __name__ == '__main__':
#     import pywt, cv2
#     from datetime import datetime
#
#     wavelet = pywt.Wavelet('bior1.1')
#     h = wavelet.rec_lo
#     g = wavelet.rec_hi
#     h_ = wavelet.dec_lo
#     g_ = wavelet.dec_hi
#     h_.reverse()
#     g_.reverse()
#
#     #"""
#     image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
#     image = cv2.imread(image_full_name, flags = 1)
#     image = image[0:512,0:512,:]
#     print(image.shape)
#     height, width, channel = image.shape
#     #image = image.reshape((1,height,width))
#     t0 = datetime.now()
#     for index in range(1):
#         m0 = DWT_2D(wavename = 'haar')
#
#         m1 = IDWT_2D(wavename = 'haar')
#     print(isinstance(m1, IDWT_2D))
#     t1 = datetime.now()

class SegNet_VGG(nn.Module):
    def __init__(self, features, num_classes = 21, init_weights = True, wavename = None):
        super(SegNet_VGG, self).__init__()
        self.features = features[0]
        self.decoders = features[1]
        self.classifier_seg = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            #nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size = 1, padding = 0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        xx = self.features(x)
        x, [(indices_1,), (indices_2,), (indices_3,), (indices_4,), (indices_5,)] = xx
        x = self.decoders(x, indices_5, indices_4, indices_3, indices_2, indices_1)
        x = self.classifier_seg(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return 'SegNet_VGG'


class WSegNet_VGG(nn.Module):
    def __init__(self, features, num_classes, init_weights = True, wavename = None):
        super(WSegNet_VGG, self).__init__()
        self.features = features[0]
        self.decoders = features[1]
        self.classifier_seg = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            #nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size = 1, padding = 0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        xx = self.features(x)
        x, [(LH1,HL1,HH1), (LH2,HL2,HH2,), (LH3,HL3,HH3,), (LH4,HL4,HH4,), (LH5,HL5,HH5,)] = xx
        x = self.decoders(x, LH5,HL5,HH5, LH4,HL4,HH4, LH3,HL3,HH3, LH2,HL2,HH2, LH1,HL1,HH1)
        x = self.classifier_seg(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return 'WSegNet_VGG'


def make_layers(cfg, batch_norm = False):
    encoder = []
    in_channels = 3
    for v in cfg:
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                encoder += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                encoder += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            encoder += [nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [nn.MaxUnpool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(v, out_channels, kernel_size = 3, padding = 1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace = True)]
    decoder = My_Sequential_re(*decoder)
    return encoder, decoder


def make_w_layers(cfg, in_channels, batch_norm = False, wavename = 'haar'):
    encoder = []
    for v in cfg:
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                encoder += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                encoder += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            encoder += [DWT_2D(wavename = wavename)]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [IDWT_2D(wavename = wavename)]
        else:
            conv2d = nn.Conv2d(v, out_channels, kernel_size = 3, padding = 1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace = True)]
    decoder = My_Sequential_re(*decoder)
    return encoder, decoder


cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],   # 11 layers
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],   # 13 layers
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],   # 16 layers out_channels for encoder, input_channels for decoder
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],   # 19 layers
}

def segnet_vgg11(pretrained = False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['A']), **kwargs)
    return model


def segnet_vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['A'], batch_norm = True), **kwargs)
    return model


def segnet_vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['B']), **kwargs)
    return model


def segnet_vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def segnet_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['D']), **kwargs)
    return model


def segnet_vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def segnet_vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['E']), **kwargs)
    return model


def segnet_vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

"""================================================================================="""

def wsegnet_vgg11(pretrained = False, wavename = 'haar', **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['A'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg11_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['A'], batch_norm = True, wavename = wavename), **kwargs)
    return model


def wsegnet_vgg13(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['B'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg13_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['B'], batch_norm=True, wavename = wavename), **kwargs)
    return model


def wsegnet_vgg16(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['D'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg16_bn(in_channels, num_classes, pretrained=False, wavename = 'haar', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['D'], in_channels, batch_norm=True, wavename = wavename), num_classes, **kwargs)
    return model


def wsegnet_vgg19(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['E'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg19_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['E'], batch_norm=True, wavename = wavename), **kwargs)
    return model


# if __name__ == '__main__':
#     from loss.loss_function import segmentation_loss
#     criterion = segmentation_loss('dice', False)
#     mask = torch.ones(2, 128, 128).long()
#     model = wsegnet_vgg16_bn(1, 5)
#     model.train()
#     input1 = torch.rand(2, 1, 128, 128)
#     y = model(input1)
#     loss_train = criterion(y, mask)
#     loss_train.backward()
#     # print(output)
#     print(y.data.cpu().numpy().shape)
#     print(loss_train)