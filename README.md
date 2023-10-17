
# XNet: Wavelet-Based Low and High Frequency Merging Networks for Semi- and Supervised Semantic Segmentation of Biomedical Images

This is the official code of [XNet: Wavelet-Based Low and High Frequency Merging Networks for Semi- and Supervised Semantic Segmentation of Biomedical Images](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_XNet_Wavelet-Based_Low_and_High_Frequency_Fusion_Networks_for_Fully-_ICCV_2023_paper.html) (ICCV 2023).

## Overview
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/Architecture%20of%20XNet.png" width="100%" ></img>
<br>Architecture of XNet.
</p>
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/visualize%20LF%20and%20HF%20images.png" width="100%" ></img>
<br>Visualize dual-branch inputs. (a) Raw image. (b) Wavelet transform results. (c) Low frequency image. (d) High frequency image.
</p>

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/Architecture%20of%20LF%20and%20HF%20fusion%20module.png" width="50%" ></img>
<br>Architecture of LF and HF fusion module.
</p>


## Quantitative Comparison

Comparison with fully- and semi-supervised state-of-the-art models on GlaS and CREMI test set. Semi-supervised models are based on UNet. DS indicates deep supervision. * indicates lightweight models. ‡ indicates training for 1000 epochs. - indicates training failed. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/Comparison%20results%20on%20GlaS%20and%20CREMI.png" width="100%" >
</p>

Comparison with fully- and semi-supervised state-of-the-art models on LA and LiTS test set. Due to GPU memory limitations, some semi-supervised models using smaller architectures, ✝ and * indicate models are based on lightweight 3D UNet (half of channels) and VNet, respectively. ‡ indicates training for 1000 epochs. - indicates training failed. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/Comparison%20results%20on%20LA%20and%20P-CT.png" width="100%" >
</p>

## Qualitative Comparison

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/XNet/blob/main/figure/Qualitative%20results.png" width="100%" >
<br>Qualitative results on GIaS, CREMI, LA and LiTS. (a) Raw images. (b) Ground truth. (c) MT. (d) Semi-supervised XNet (3D XNet). (e) UNet (3D UNet). (f) Fully-Supervised XNet (3D XNet). The orange arrows highlight the difference among of the results.
</p>

## Reimplemented Architecture
We have reimplemented some 2D and 3D models in semi- and supervised semantic segmentation.
<table>
<tr><th align="left">Method</th> <th align="left">Dimension</th><th align="left">Model</th><th align="left">Code</th></tr>
<tr><td rowspan="23">Supervised</td> <td rowspan="13">2D</td><td>UNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/unet.py">models/networks_2d/unet.py</a></td></tr>
<tr><td>UNet++</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/unet_plusplus.py">models/networks_2d/unet_plusplus.py</a></td></tr>
<tr><td>Att-UNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/unet.py">models/networks_2d/unet.py</a></td></tr>
<tr><td>Aerial LaneNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/aerial_lanenet.py">models/networks_2d/aerial_lanenet.py</a></td></tr>
<tr><td>MWCNN</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/mwcnn.py">models/networks_2d/mwcnn.py</a></td></tr>
<tr><td>HRNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/hrnet.py">models/networks_2d/hrnet.py</a></td></tr>
<tr><td>Res-UNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/resunet.py">models/networks_2d/resunet.py</a></td></tr>
<tr><td>WDS</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/wds.py">models/networks_2d/wds.py</a></td></tr>
<tr><td>U<sup>2</sup>-Net</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/u2net.py">models/networks_2d/u2net.py</a></td></tr>
<tr><td>UNet 3+</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/unet_3plus.py">models/networks_2d/unet_3plus.py</a></td></tr>
<tr><td>SwinUNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/swinunet.py">models/networks_2d/swinunet.py</a></td></tr>
<tr><td>WaveSNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/wavesnet.py">models/networks_2d/wavesnet.py</a></td></tr>
<tr><td>XNet (Ours)</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_2d/xnet.py">models/networks_2d/xnet.py</a></td></tr>
<tr><td rowspan="10">3D</td><td>VNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/vnet.py">models/networks_3d/vnet.py</a></td></tr>
<tr><td>UNet 3D</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/unet3d.py">models/networks_3d/unet3d.py</a></td></tr>
<tr><td>Res-UNet 3D</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/res_unet3d.py">models/networks_3d/res_unet3d.py</a></td></tr>
<tr><td>ESPNet 3D</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/espnet3d.py">models/networks_3d/espnet3d.py</a></td></tr>
<tr><td>DMFNet 3D</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/dmfnet.py">models/networks_3d/dmfnet.py</a></td></tr>
<tr><td>ConResNet</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/conresnet.py">models/networks_3d/conresnet.py</a></td></tr>
<tr><td>CoTr</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/cotr.py">models/networks_3d/cotr.py</a></td></tr>
<tr><td>TransBTS</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/transbts.py">models/networks_3d/transbts.py</a></td></tr>
<tr><td>UNETR</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/unetr.py">models/networks_3d/unetr.py</a></td></tr>
<tr><td>XNet 3D (Ours)</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/models/networks_3d/xnet3d.py">models/networks_3d/xnet3d.py</a></td></tr>
<tr><td rowspan="17">Semi-Supervised</td> <td rowspan="8">2D</td><td>MT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_MT.py">train_semi_MT.py</a></td></tr>
<tr><td>EM</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_EM.py">train_semi_EM.py</a></td></tr>
<tr><td>UAMT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_UAMT.py">train_semi_UAMT.py</a></td></tr>
<tr><td>CCT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CCT.py">train_semi_CCT.py</a></td></tr>
<tr><td>CPS</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CPS.py">train_semi_CPS.py</a></td></tr>
<tr><td>URPC</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_URPC.py">train_semi_URPC.py</a></td></tr>
<tr><td>CT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CT.py">train_semi_CT.py</a></td></tr>
<tr><td>XNet (Ours)</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_XNet.py">train_semi_XNet.py</a></td></tr>
<td rowspan="9">3D</td><td>MT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_MT_3d.py">train_semi_MT_3d.py</a></td></tr>
<tr><td>EM</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_EM_3d.py">train_semi_EM_3d.py</a></td></tr>
<tr><td>UAMT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_UAMT_3d.py">train_semi_UAMT_3d.py</a></td></tr>
<tr><td>CCT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CCT_3d.py">train_semi_CCT_3d.py</a></td></tr>
<tr><td>CPS</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CPS_3d.py">train_semi_CPS_3d.py</a></td></tr>
<tr><td>URPC</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_URPC_3d.py">train_semi_URPC_3d.py</a></td></tr>
<tr><td>CT</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_CT_3d.py">train_semi_CT_3d.py</a></td></tr>
<tr><td>DTC</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_DTC.py">train_semi_DTC.py</a></td></tr>
<tr><td>XNet 3D (Ours)</td><td><a href="https://github.com/Yanfeng-Zhou/XNet/blob/main/train_semi_XNet3d.py">train_semi_XNet3d.py</a></td></tr>
</table>

## Requirements
```
albumentations==0.5.2
einops==0.4.1
MedPy==0.4.0
numpy==1.20.2
opencv_python==4.2.0.34
opencv_python_headless==4.5.1.48
Pillow==8.0.0
PyWavelets==1.1.1
scikit_image==0.18.1
scikit_learn==1.0.1
scipy==1.4.1
SimpleITK==2.1.0
timm==0.6.7
torch==1.8.0+cu111
torchio==0.18.53
torchvision==0.9.0+cu111
tqdm==4.65.0
visdom==0.1.8.9
```

## Usage
**Data preparation**
Your datasets directory tree should be look like this:
>to see [tools/wavelet2D.py](https://github.com/Yanfeng-Zhou/XNet/blob/main/tools/wavelet2D.py) and  [tools/wavelet3D.py](https://github.com/Yanfeng-Zhou/XNet/blob/main/tools/wavelet3D.py) for **L** and **H**
```
dataset
├── train_sup_100
    ├── L
        ├── 1.tif
        ├── 2.tif
        └── ...
    ├── H
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
├── train_sup_20
    ├── L
    ├── H
    └── mask
├── train_unsup_80
    └── L
    ├── H
└── val
    ├── L
    ├── H
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
@InProceedings{Zhou_2023_ICCV,
  author = {Zhou, Yanfeng and Huang, Jiaxing and Wang, Chenlong and Song, Le and Yang, Ge}, 
  title = {XNet: Wavelet-Based Low and High Frequency Fusion Networks for Fully- and Semi-Supervised Semantic Segmentation of Biomedical Images}, 
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  month = {October}, 
  year = {2023}, 
  pages = {21085-21096}
  }
```



