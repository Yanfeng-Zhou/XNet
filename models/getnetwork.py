import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'xnet':
        net = XNet(in_channels, num_classes)
    elif network == 'xnet_sb':
        net = XNet_sb(in_channels, num_classes)
    elif network == 'xnet_1_1_m':
        net = XNet_1_1_m(in_channels, num_classes)
    elif network == 'xnet_1_2_m':
        net = XNet_1_2_m(in_channels, num_classes)
    elif network == 'xnet_2_1_m':
        net = XNet_2_1_m(in_channels, num_classes)
    elif network == 'xnet_3_2_m':
        net = XNet_3_2_m(in_channels, num_classes)
    elif network == 'xnet_2_3_m':
        net = XNet_2_3_m(in_channels, num_classes)
    elif network == 'xnet_3_3_m':
        net = XNet_3_3_m(in_channels, num_classes)
    elif network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_plusplus' or network == 'unet++':
        net = unet_plusplus(in_channels, num_classes)
    elif network == 'r2unet':
        net = r2_unet(in_channels, num_classes)
    elif network == 'attunet':
        net = attention_unet(in_channels, num_classes)
    elif network == 'hrnet18':
        net = hrnet18(in_channels, num_classes)
    elif network == 'hrnet48':
        net = hrnet48(in_channels, num_classes)
    elif network == 'resunet':
        net = res_unet(in_channels, num_classes)
    elif network == 'resunet++':
        net = res_unet_plusplus(in_channels, num_classes)
    elif network == 'u2net':
        net = u2net(in_channels, num_classes)
    elif network == 'u2net_s':
        net = u2net_small(in_channels, num_classes)
    elif network == 'unet3+':
        net = unet_3plus(in_channels, num_classes)
    elif network == 'unet3+_ds':
        net = unet_3plus_ds(in_channels, num_classes)
    elif network == 'unet3+_ds_cgm':
        net = unet_3plus_ds_cgm(in_channels, num_classes)
    elif network == 'swinunet':
        net = swinunet(num_classes, 224)  # img_size = 224
    elif network == 'unet_urpc':
        net = unet_urpc(in_channels, num_classes)
    elif network == 'unet_cct':
        net = unet_cct(in_channels, num_classes)
    elif network == 'wavesnet':
        net = wsegnet_vgg16_bn(in_channels, num_classes)
    elif network == 'mwcnn':
        net = mwcnn(in_channels, num_classes)
    elif network == 'alnet':
        net = Aerial_LaneNet(in_channels, num_classes)
    elif network == 'wds':
        net = WDS(in_channels, num_classes)

    # 3d networks
    elif network == 'xnet3d':
        net = xnet3d(in_channels, num_classes)
    elif network == 'unet3d':
        net = unet3d(in_channels, num_classes)
    elif network == 'unet3d_min':
        net = unet3d_min(in_channels, num_classes)
    elif network == 'unet3d_urpc':
        net = unet3d_urpc(in_channels, num_classes)
    elif network == 'unet3d_cct':
        net = unet3d_cct(in_channels, num_classes)
    elif network == 'unet3d_cct_min':
        net = unet3d_cct_min(in_channels, num_classes)
    elif network == 'unet3d_dtc':
        net = unet3d_dtc(in_channels, num_classes)
    elif network == 'vnet':
        net = vnet(in_channels, num_classes)
    elif network == 'vnet_cct':
        net = vnet_cct(in_channels, num_classes)
    elif network == 'vnet_dtc':
        net = vnet_dtc(in_channels, num_classes)
    elif network == 'resunet3d':
        net = res_unet3d(in_channels, num_classes)
    elif network == 'conresnet':
        net = conresnet(in_channels, num_classes, img_shape=kwargs['img_shape'])
    elif network == 'espnet3d':
        net = espnet3d(in_channels, num_classes)
    elif network == 'dmfnet':
        net = dmfnet(in_channels, num_classes)
    elif network == 'transbts':
        net = transbts(in_channels, num_classes, img_shape=kwargs['img_shape'])
    elif network == 'cotr':
        net = cotr(in_channels, num_classes)
    elif network == 'unertr':
        net = unertr(in_channels, num_classes, img_shape=kwargs['img_shape'])
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
