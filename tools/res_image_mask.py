import numpy as np
import os
import argparse
import SimpleITK as sitk

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='//10.0.5.233/shared_data/XNet/dataset/Atrial/train_sup_100')
    args = parser.parse_args()

    image_path = args.data_path + '/image'
    mask_path = args.data_path + '/mask'

    save_res_path = args.data_path + '/image_res'
    save_res_mask_path = args.data_path + '/mask_res'
    if not os.path.exists(save_res_path):
        os.mkdir(save_res_path)
    if not os.path.exists(save_res_mask_path):
        os.mkdir(save_res_mask_path)

    for i in os.listdir(image_path):

        image = sitk.ReadImage(os.path.join(image_path, i))
        image_np = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(os.path.join(mask_path, i))
        mask_np = sitk.GetArrayFromImage(mask)

        image_copy = np.zeros(image_np.shape)
        image_copy[1:, :, :] = image_np[0:image_np.shape[0] - 1, :, :]
        image_res = image_np - image_copy
        image_res[0, :, :] = 0
        image_res = np.abs(image_res)
        image_res = sitk.GetImageFromArray(image_res)
        image_res.SetSpacing(image.GetSpacing())
        image_res.SetDirection(image.GetDirection())
        image_res.SetOrigin(image.GetOrigin())

        mask_copy = np.zeros(mask_np.shape)
        mask_copy[1:, :, :] = mask_np[0:mask_np.shape[0] - 1, :, :]
        mask_res = mask_np - mask_copy
        mask_res[0, :, :] = 0
        mask_res = np.abs(mask_res)
        mask_res = sitk.GetImageFromArray(mask_res)
        mask_res.SetSpacing(image.GetSpacing())
        mask_res.SetDirection(image.GetDirection())
        mask_res.SetOrigin(image.GetOrigin())

        sitk.WriteImage(image_res, os.path.join(save_res_path, i))
        sitk.WriteImage(mask_res, os.path.join(save_res_mask_path, i))



