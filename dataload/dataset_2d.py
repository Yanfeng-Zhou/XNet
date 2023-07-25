import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np


class dataset_itn(Dataset):
    def __init__(self, data_dir, input1, augmentation_1, normalize_1, sup=True, num_images=None, **kwargs):
        super(dataset_itn, self).__init__()

        img_paths_1 = []
        mask_paths = []

        image_dir_1 = data_dir + '/' + input1
        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_1):

            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img_1 = Image.open(img_path_1)
        img_1 = np.array(img_1)

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment_1 = self.augmentation_1(image=img_1, mask=mask)
            img_1 = augment_1['image']
            mask_1 = augment_1['mask']

            normalize_1 = self.normalize_1(image=img_1, mask=mask_1)
            img_1 = normalize_1['image']
            mask_1 = normalize_1['mask']
            mask_1 = mask_1.long()

            sampel = {'image': img_1, 'mask': mask_1, 'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img_1)
            img_1 = augment_1['image']
            normalize_1 = self.normalize_1(image=img_1)
            img_1 = normalize_1['image']

            sampel = {'image': img_1, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_itn(data_dir, input1, data_transform_1, data_normalize_1, sup=True, num_images=None, **kwargs):
    dataset = dataset_itn(data_dir=data_dir,
                           input1=input1,
                           augmentation_1=data_transform_1,
                           normalize_1=data_normalize_1,
                           sup=sup,
                           num_images=num_images,
                           **kwargs
                           )
    return dataset


class dataset_iitnn(Dataset):
    def __init__(self, data_dir, input1, input2, augmentation1, normalize_1, normalize_2, sup=True,
                 num_images=None, **kwargs):
        super(dataset_iitnn, self).__init__()

        img_paths_1 = []
        img_paths_2 = []
        mask_paths = []

        image_dir_1 = data_dir + '/' + input1
        image_dir_2 = data_dir + '/' + input2
        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_1):

            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            image_path_2 = os.path.join(image_dir_2, image)
            img_paths_2.append(image_path_2)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        assert len(img_paths_1) == len(img_paths_2)
        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
                img_paths_2 = img_paths_2[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]
                img_paths_2 = img_paths_2 * quotient
                img_paths_2 += [img_paths_2[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.img_paths_2 = img_paths_2
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation1
        self.normalize_1 = normalize_1
        self.normalize_2 = normalize_2
        self.sup = sup
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img_1 = Image.open(img_path_1)
        img_1 = np.array(img_1)

        img_path_2 = self.img_paths_2[index]
        img_2 = Image.open(img_path_2)
        img_2 = np.array(img_2)

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment_1 = self.augmentation_1(image=img_1, image2=img_2, mask=mask)
            img_1 = augment_1['image']
            img_2 = augment_1['image2']
            mask = augment_1['mask']

            normalize_1 = self.normalize_1(image=img_1, mask=mask)
            img_1 = normalize_1['image']
            mask = normalize_1['mask']
            mask = mask.long()

            normalize_2 = self.normalize_2(image=img_2)
            img_2 = normalize_2['image']

            sampel = {'image': img_1, 'image_2': img_2, 'mask': mask, 'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img_1, image2=img_2)
            img_1 = augment_1['image']
            img_2 = augment_1['image2']

            normalize_1 = self.normalize_1(image=img_1)
            img_1 = normalize_1['image']

            normalize_2 = self.normalize_2(image=img_2)
            img_2 = normalize_2['image']

            sampel = {'image': img_1, 'image_2': img_2, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_iitnn(data_dir, input1, input2, data_transform_1, data_normalize_1, data_normalize_2, sup=True, num_images=None, **kwargs):
    dataset = dataset_iitnn(data_dir=data_dir,
                           input1=input1,
                           input2=input2,
                           augmentation1=data_transform_1,
                           normalize_1=data_normalize_1,
                           normalize_2=data_normalize_2,
                           sup=sup,
                           num_images=num_images,
                           **kwargs
                           )
    return dataset
