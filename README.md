
# XNet: Wavelet-Based Low and High Frequency Merging Networks for Semi- and Supervised Semantic Segmentation of Biomedical Images

This is the official code of [XNet: Wavelet-Based Low and High Frequency Merging Networks for Semi- and Supervised Semantic Segmentation of Biomedical Images](https://) (ICCV 2023).

## Overview
<p align="center">
<img src="https://i.postimg.cc/Mptz9DBJ/figure-1.png#pic_center" width="100%" ></img>
<center>Architecture of XNet</center>
</p>
<p>
<img src="https://i.postimg.cc/W1ksvkhh/figure-2.png#pic_center" width="100%" >
<center>Visualize dual-branch inputs. (a) Raw image. (b) Wavelet transform results. (c) Low frequency image. (d) High frequency image.</center>
</p>

<p align="center">
<img src="https://i.postimg.cc/mrW1fR2W/figure-3-v2.png#pic_center" width="50%" >
<center>LF and HF fusion module</center>
</p>


## Quantitative Comparison

Comparison with fully- and semi-supervised state-of-the-art models on GlaS and CREMI test set. DS indicates deep supervision. * indicates lightweight models. - indicates training failed. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.

<p align="center">
<img src="https://i.postimg.cc/zG4hpKR7/2D.png#pic_center" width="100%" >
</p>

Comparison with fully- and semi-supervised state-of-the-art models on LA and LiTS test set. Due to GPU memory limitations, some semi-supervised models using smaller architectures, ✝ and * indicate models are based on lightweight 3D UNet (half of channels) and VNet, respectively. - indicates training failed. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.

<p align="center">
<img src="https://i.postimg.cc/G2Xhgn5R/3D.png#pic_center" width="100%" >
</p>

## Qualitative Comparison

<p align="center">
<img src="https://i.postimg.cc/4xTq9w6G/figure-5.png#pic_center" width="100%" >
<center>Qualitative results on GIaS, CREMI, LA and LiTS. (a) Raw images. (b) Ground truth. (c) MT. (d) Semi-supervised XNet (3D XNet). (e) UNet (3D UNet). (f) Fully-Supervised XNet (3D XNet). The orange arrows highlight the difference among of the results.</center>
</p>

## Reimplemented Architecture
We have reimplemented some 2D and 3D models in semi- and supervised semantic segmentation.
<table>
<tr><th align="left">Method</th> <th align="left">Dimension</th><th align="left">Model</th><th align="left">Code</th></tr>
<tr><td rowspan="23">Supervised</td> <td rowspan="13">2D</td><td><a href="#">UNet</a></td><td><a href="#">models/networks_2d/unet.py</a></td></tr>
<tr><td><a href="#">UNet++</a></td><td><a href="#">models/networks_2d/unet_plusplus.py</a></td></tr>
<tr><td><a href="#">Att-UNet</a></td><td><a href="#">models/networks_2d/unet.py</a></td></tr>
<tr><td><a href="#">Aerial LaneNet</a></td><td><a href="#">models/networks_2d/aerial_lanenet.py</a></td></tr>
<tr><td><a href="#">MWCNN</a></td><td><a href="#">models/networks_2d/mwcnn.py</a></td></tr>
<tr><td><a href="#">HRNet</a></td><td><a href="#">models/networks_2d/hrnet.py</a></td></tr>
<tr><td><a href="#">Res-UNet</a></td><td><a href="#">models/networks_2d/resunet.py</a></td></tr>
<tr><td><a href="#">WDS</a></td><td><a href="#">models/networks_2d/wds.py</a></td></tr>
<tr><td><a href="#">U<sup>2</sup>-Net</a></td><td><a href="#">models/networks_2d/u2net.py</a></td></tr>
<tr><td><a href="#">UNet 3+</a></td><td><a href="#">models/networks_2d/unet_3plus.py</a></td></tr>
<tr><td><a href="#">SwinUNet</a></td><td><a href="#">models/networks_2d/swinunet.py</a></td></tr>
<tr><td><a href="#">WaveSNet</a></td><td><a href="#">models/networks_2d/wavesnet.py</a></td></tr>
<tr><td>XNet (Ours)</td><td><a href="#">models/networks_2d/xnet.py</a></td></tr>
<tr><td rowspan="10">3D</td><td><a href="#">VNet</a></td><td><a href="#">models/networks_3d/vnet.py</a></td></tr>
<tr><td><a href="#">UNet 3D</a></td><td><a href="#">models/networks_3d/unet3d.py</a></td></tr>
<tr><td>Res-UNet 3D</td><td><a href="#">models/networks_3d/res_unet3d.py</a></td></tr>
<tr><td><a href="#">ESPNet 3D</a></td><td><a href="#">models/networks_3d/espnet3d.py</a></td></tr>
<tr><td><a href="#">DMFNet 3D</a></td><td><a href="#">models/networks_3d/dmfnet.py</a></td></tr>
<tr><td><a href="#">ConResNet</a></td><td><a href="#">models/networks_3d/conresnet.py</a></td></tr>
<tr><td><a href="#">CoTr</a></td><td><a href="#">models/networks_3d/cotr.py</a></td></tr>
<tr><td><a href="#">TransBTS</a></td><td><a href="#">models/networks_3d/transbts.py</a></td></tr>
<tr><td><a href="#">UNETR</a></td><td><a href="#">models/networks_3d/unetr.py</a></td></tr>
<tr><td>XNet 3D (Ours)</td><td><a href="#">models/networks_3d/xnet3d.py</a></td></tr>
<tr><td rowspan="17">Semi-Supervised</td> <td rowspan="8">2D</td><td><a href="#">MT</a></td><td><a href="#">train_semi_MT.py</a></td></tr>
<tr><td><a href="#">EM</a></td><td><a href="#">train_semi_EM.py</a></td></tr>
<tr><td><a href="#">UAMT</a></td><td><a href="#">train_semi_UAMT.py</a></td></tr>
<tr><td><a href="#">CCT</a></td><td><a href="#">train_semi_CCT.py</a></td></tr>
<tr><td><a href="#">CPS</a></td><td><a href="#">train_semi_CPS.py</a></td></tr>
<tr><td><a href="#">URPC</a></td><td><a href="#">train_semi_URPC.py</a></td></tr>
<tr><td><a href="#">CT</a></td><td><a href="#">train_semi_CT.py</a></td></tr>
<tr><td>XNet (Ours)</td><td><a href="#">train_semi_XNet.py</a></td></tr>
<td rowspan="9">3D</td><td><a href="#">MT</a></td><td><a href="#">train_semi_MT_3d.py</a></td></tr>
<tr><td><a href="#">EM</a></td><td><a href="#">train_semi_EM_3d.py</a></td></tr>
<tr><td><a href="#">UAMT</a></td><td><a href="#">train_semi_UAMT_3d.py</a></td></tr>
<tr><td><a href="#">CCT</a></td><td><a href="#">train_semi_CCT_3d.py</a></td></tr>
<tr><td><a href="#">CPS</a></td><td><a href="#">train_semi_CPS_3d.py</a></td></tr>
<tr><td><a href="#">URPC</a></td><td><a href="#">train_semi_URPC_3d.py</a></td></tr>
<tr><td><a href="#">CT</a></td><td><a href="#">train_semi_CT_3d.py</a></td></tr>
<tr><td><a href="#">DTC</a></td><td><a href="#">train_semi_DTC.py</a></td></tr>
<tr><td>XNet 3D (Ours)</td><td><a href="#">train_semi_XNet3d.py</a></td></tr>
</table>

## Requirements
```
albumentations==1.2.1
einops==0.4.1
MedPy==0.4.0
numpy==1.21.5
opencv_python_headless==4.5.4.60
Pillow==9.2.0
PyWavelets==1.3.0
scikit_image==0.19.3
scikit_learn==1.1.2
scipy==1.7.3
SimpleITK==2.2.0
skimage==0.0
timm==0.6.7
torch==1.8.0+cu111
torchio==0.18.84
torchvision==0.9.0+cu111
tqdm==4.64.0
visdom==0.1.8.9
```

## Usage
**Data preparation**
Your datasets directory tree should be look like this:
>to see [tools/wavelet2D.py](https://) and  [tools/wavelet3D.py](https://) for **DB2_H**
```
dataset
├── train_sup_100
    ├── image
        ├── 1.tif
        ├── 2.tif
        └── ...
    ├── DB2_H
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
├── train_sup_20
    ├── image
    ├── DB2_H
    └── mask
├── train_unsup_80
    └── image
    ├── DB2_H
└── val
    ├── image
    ├── DB2_H
    └── mask
```
**Supervised training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_sup_XNet.py
```
**Semi-supervised training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_semi_XNet.py
```
**Testing**
```
python -m torch.distributed.launch --nproc_per_node=4 test.py
```

## Citation
If our work is useful for your research, please cite our paper:
```
```

