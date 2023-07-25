import numpy as np
import os
import argparse
from tqdm import tqdm
import SimpleITK as sitk

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='E:/Biomedical datasets/LiTS')
    parser.add_argument('--save_path', default='E:/Biomedical datasets/LiTS/dataset')
    parser.add_argument('--min_hu', default=-100)
    parser.add_argument('--max_hu', default=250)
    parser.add_argument('--target_spacing', default=[1.00, 1.00, 1.00])
    parser.add_argument('--crop_pixel', default=25)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_image_path = args.save_path + '/image'
    save_mask_path = args.save_path + '/mask'
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    image_path = args.data_path + '/image'
    mask_path = args.data_path + '/mask'

    for i in os.listdir(image_path):

        image_dir = os.path.join(image_path, i)
        mask_dir = os.path.join(mask_path, i)

        image = sitk.ReadImage(image_dir)
        mask = sitk.ReadImage(mask_dir)

        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        new_size = size * spacing / args.target_spacing
        new_size = [int(s) for s in new_size]

        print(new_size, size)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetOutputDirection(image.GetDirection())
        resample_image.SetOutputOrigin(image.GetOrigin())
        resample_image.SetSize(new_size)
        resample_image.SetOutputSpacing(args.target_spacing)
        resample_image.SetInterpolator(sitk.sitkLinear)
        image = resample_image.Execute(image)

        resample_mask = sitk.ResampleImageFilter()
        resample_mask.SetOutputDirection(mask.GetDirection())
        resample_mask.SetOutputOrigin(mask.GetOrigin())
        resample_mask.SetSize(new_size)
        resample_mask.SetOutputSpacing(args.target_spacing)
        resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = resample_mask.Execute(mask)

        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)

        w, h, d = mask_np.shape
        templ = np.nonzero(mask_np)
        w_min = max(np.min(templ[0]) - args.crop_pixel, 0)
        w_max = min(np.max(templ[0]) + args.crop_pixel, w)
        h_min = max(np.min(templ[1]) - args.crop_pixel, 0)
        h_max = min(np.max(templ[1]) + args.crop_pixel, h)
        d_min = max(np.min(templ[2]) - args.crop_pixel, 0)
        d_max = min(np.max(templ[2]) + args.crop_pixel, d)

        image_np = image_np[w_min:w_max, h_min:h_max, d_min:d_max]
        # image_np = image.data
        image_np[image_np < args.min_hu] = args.min_hu
        image_np[image_np > args.max_hu] = args.max_hu

        mask_np = mask_np[w_min:w_max, h_min:h_max, d_min:d_max]


        image_save = sitk.GetImageFromArray(image_np)
        image_save.SetSpacing(args.target_spacing)
        image_save.SetDirection(image.GetDirection())
        image_save.SetOrigin(image.GetOrigin())

        mask_save = sitk.GetImageFromArray(mask_np)
        mask_save.SetSpacing(args.target_spacing)
        mask_save.SetDirection(image.GetDirection())
        mask_save.SetOrigin(image.GetOrigin())

        sitk.WriteImage(image_save, os.path.join(save_image_path, i))
        sitk.WriteImage(mask_save, os.path.join(save_mask_path, i))
        # image_save.save(os.path.join(save_image_path, save_name))
        # mask_save.save(os.path.join(save_mask_path, save_name))


