import numpy as np
import os
import argparse
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage import segmentation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/val')
    parser.add_argument('--num_classes', default=3)
    args = parser.parse_args()

    mask_path = args.data_path + '/mask'

    for i in range(args.num_classes-1):

        save_sdf_mask_path = args.data_path + '/mask_sdf' + str(i+1)
        if not os.path.exists(save_sdf_mask_path):
            os.mkdir(save_sdf_mask_path)

        for j in os.listdir(mask_path):

            mask = sitk.ReadImage(os.path.join(mask_path, j))
            mask_np = sitk.GetArrayFromImage(mask)

            mask_np[mask_np != (i+1)] = 0
            mask_np = mask_np.astype(bool)
            if mask_np.any():
                mask_neg = ~mask_np
                posdis = distance_transform_edt(mask_np)
                negdis = distance_transform_edt(mask_neg)
                boundary = segmentation.find_boundaries(mask_np, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary == 1] = 0
                # sdf = ((sdf - np.min(sdf)) / (np.max(sdf) - np.min(sdf))) * 255
            else:
                sdf = np.zeros(mask_np.shape)

            sdf = sitk.GetImageFromArray(sdf)
            sdf.SetSpacing(mask.GetSpacing())
            sdf.SetDirection(mask.GetDirection())
            sdf.SetOrigin(mask.GetOrigin())
            sitk.WriteImage(sdf, os.path.join(save_sdf_mask_path, j))






