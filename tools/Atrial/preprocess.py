import numpy as np
import torchio as tio
import os
import argparse
from tqdm import tqdm
import SimpleITK as sitk

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='E:/Biomedical datasets/2018 Atrial Segmentation Challenge/Training Set')
    parser.add_argument('--save_path', default='E:/Biomedical datasets/2018 Atrial Segmentation Challenge/dataset')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_image_path = args.save_path + '/image'
    save_mask_path = args.save_path + '/mask'
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    for i in os.listdir(args.data_path):
        save_name = i + '.nrrd'
        image_path = args.data_path + '/' + i + '/' + 'lgemri.nrrd'
        mask_path = args.data_path + '/' + i + '/' + 'laendo.nrrd'

        image = tio.ScalarImage(image_path)
        mask = tio.LabelMap(mask_path)

        _, w, h, d = mask.data.shape
        tempL = np.nonzero(np.array(mask.data))
        minx, maxx = np.min(tempL[1]), np.max(tempL[1])
        miny, maxy = np.min(tempL[2]), np.max(tempL[2])
        # minz, maxz = np.min(tempL[3]), np.max(tempL[3])

        px = max(112 - (maxx - minx), 0) // 2
        py = max(112 - (maxy - miny), 0) // 2
        # pz = max(80 - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        # minz = max(minz - np.random.randint(5, 10) - pz, 0)
        # maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image_np = image.data[:, minx:maxx, miny:maxy, :]
        image.set_data(image_np)

        mask_np = mask.data[:, minx:maxx, miny:maxy, :]
        mask.set_data(mask_np)

        print(image_np.shape)
        image.save(os.path.join(save_image_path, save_name))
        mask.save(os.path.join(save_mask_path, save_name))


