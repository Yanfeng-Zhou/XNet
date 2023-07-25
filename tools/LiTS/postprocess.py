import numpy as np
import argparse
import os
import SimpleITK as sitk
from skimage.morphology import remove_small_objects, remove_small_holes
import skimage

def save_max_objects(image):
    labeled_image = skimage.measure.label(image)
    labeled_list = skimage.measure.regionprops(labeled_image)
    box = []
    for i in range(len(labeled_list)):
        box.append(labeled_list[i].area)
        label_num = box.index(max(box)) + 1

    labeled_image[labeled_image != label_num] = 0
    labeled_image[labeled_image == label_num] = 1

    return labeled_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='//10.0.5.233/shared_data/XNet/seg_pred/test/LiTS/best_DTC_Jc_0.7594')
    parser.add_argument('--save_path', default='//10.0.5.233/shared_data/XNet/seg_pred/test/LiTS/best_DTC_Jc_0.7594_mor')
    parser.add_argument('--fill_hole_thr', default=100)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for i in os.listdir(args.pred_path):

        pred_path = os.path.join(args.pred_path, i)
        save_path = os.path.join(args.save_path, i)

        pred = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(pred)

        pred_ = pred.copy()
        pred_[pred != 0] = 1

        pred_ = pred_.astype(bool)
        pred_ = remove_small_holes(pred_, args.fill_hole_thr)
        pred_ = pred_.astype(np.uint8)

        pred_ = save_max_objects(pred_)
        pred_[(pred_ == 1) & (pred == 2)] = 2
        pred_ = pred_.astype(np.uint8)

        pred_ = sitk.GetImageFromArray(pred_)
        sitk.WriteImage(pred_, save_path)