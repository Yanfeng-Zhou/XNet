import numpy as np
from PIL import Image
import pywt
import argparse
import os
import SimpleITK as sitk
import torchio as tio

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/val/image')
    parser.add_argument('--L_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/L')
    parser.add_argument('--H_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/H')
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')

    args = parser.parse_args()

    if not os.path.exists(args.L_path):
        os.mkdir(args.L_path)

    if not os.path.exists(args.H_path):
        os.mkdir(args.H_path)

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        L_path = os.path.join(args.L_path, i)
        H_path = os.path.join(args.H_path, i)

        image = sitk.ReadImage(image_path)
        image_np = sitk.GetArrayFromImage(image)

        image_wave = pywt.dwtn(image_np, args.wavelet_type)
        LLL = image_wave['aaa']
        LLH = image_wave['aad']
        LHL = image_wave['ada']
        LHH = image_wave['add']
        HLL = image_wave['daa']
        HLH = image_wave['dad']
        HHL = image_wave['dda']
        HHH = image_wave['ddd']

        LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize())
        resample_image.SetOutputSpacing([0.5, 0.5, 0.5])
        resample_image.SetInterpolator(sitk.sitkLinear)
        LLL = resample_image.Execute(LLL)

        LLL.SetSpacing(image.GetSpacing())
        LLL.SetDirection(image.GetDirection())
        LLL.SetOrigin(image.GetOrigin())

        sitk.WriteImage(LLL, L_path)


        LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
        LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
        LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
        HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
        HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
        HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
        HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

        merge1 = LLH + LHL + LHH + HLL + HLH + HHL + HHH
        merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255

        merge1 = sitk.GetImageFromArray(merge1)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize())
        resample_image.SetOutputSpacing([0.5, 0.5, 0.5])
        resample_image.SetInterpolator(sitk.sitkLinear)
        merge1 = resample_image.Execute(merge1)

        merge1.SetSpacing(image.GetSpacing())
        merge1.SetDirection(image.GetDirection())
        merge1.SetOrigin(image.GetOrigin())

        sitk.WriteImage(merge1, H_path)


