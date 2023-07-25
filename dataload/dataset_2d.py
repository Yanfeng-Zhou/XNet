import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pywt

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

class dataset_wds(Dataset):
    def __init__(self, data_dir, augmentation1, normalize_LL, normalize_LH, normalize_HL, normalize_HH, **kwargs):
        super(dataset_wds, self).__init__()

        img_paths_LL = []
        img_paths_LH = []
        img_paths_HL = []
        img_paths_HH = []
        mask_paths = []
        image_dir_LL = data_dir + '/LL'
        image_dir_LH = data_dir + '/LH'
        image_dir_HL = data_dir + '/HL'
        image_dir_HH = data_dir + '/HH'
        mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_LL):

            image_path_LL = os.path.join(image_dir_LL, image)
            img_paths_LL.append(image_path_LL)
            image_path_LH = os.path.join(image_dir_LH, image)
            img_paths_LH.append(image_path_LH)
            image_path_HL = os.path.join(image_dir_HL, image)
            img_paths_HL.append(image_path_HL)
            image_path_HH = os.path.join(image_dir_HH, image)
            img_paths_HH.append(image_path_HH)

            mask_path = os.path.join(mask_dir, image)
            mask_paths.append(mask_path)

        self.img_paths_LL = img_paths_LL
        self.img_paths_LH = img_paths_LH
        self.img_paths_HL = img_paths_HL
        self.img_paths_HH = img_paths_HH
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation1
        self.normalize_LL = normalize_LL
        self.normalize_LH = normalize_LH
        self.normalize_HL = normalize_HL
        self.normalize_HH = normalize_HH
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path_LL = self.img_paths_LL[index]
        img_LL = Image.open(img_path_LL)
        img_LL = np.array(img_LL)

        img_path_LH = self.img_paths_LH[index]
        img_LH = Image.open(img_path_LH)
        img_LH = np.array(img_LH)

        img_path_HL = self.img_paths_HL[index]
        img_HL = Image.open(img_path_HL)
        img_HL = np.array(img_HL)

        img_path_HH = self.img_paths_HH[index]
        img_HH = Image.open(img_path_HH)
        img_HH = np.array(img_HH)

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = np.array(mask)

        augment_1 = self.augmentation_1(image=img_LL, mask=mask, imageLH=img_LH, imageHL=img_HL, imageHH=img_HH)
        img_LL = augment_1['image']
        img_LH = augment_1['imageLH']
        img_HL = augment_1['imageHL']
        img_HH = augment_1['imageHH']
        mask_1 = augment_1['mask']

        normalize_LL = self.normalize_LL(image=img_LL, mask=mask_1)
        img_LL = normalize_LL['image']
        mask_1 = normalize_LL['mask']
        mask_1 = mask_1.long()

        normalize_LH = self.normalize_LH(image=img_LH)
        img_LH = normalize_LH['image']

        normalize_HL = self.normalize_HL(image=img_HL)
        img_HL = normalize_HL['image']

        normalize_HH = self.normalize_HH(image=img_HH)
        img_HH = normalize_HH['image']

        sampel = {'image_LL': img_LL, 'image_LH': img_LH, 'image_HL': img_HL,  'image_HH': img_HH, 'mask': mask_1, 'ID': os.path.split(mask_path)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_LL)


def imagefloder_wds(data_dir, data_transform_1, data_normalize_LL, data_normalize_LH, data_normalize_HL, data_normalize_HH, **kwargs):
    dataset = dataset_wds(data_dir=data_dir,
                           augmentation1=data_transform_1,
                           normalize_LL=data_normalize_LL,
                           normalize_LH=data_normalize_LH,
                           normalize_HL=data_normalize_HL,
                           normalize_HH=data_normalize_HH,
                           **kwargs
                           )
    return dataset

class dataset_aerial_lanenet(Dataset):
    def __init__(self, data_dir, augmentation1, normalize_1, normalize_l1, normalize_l2, normalize_l3, normalize_l4, **kwargs):
        super(dataset_aerial_lanenet, self).__init__()

        img_paths = []
        mask_paths = []
        image_dir = data_dir + '/image'
        mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir):

            image_path = os.path.join(image_dir, image)
            img_paths.append(image_path)

            mask_path = os.path.join(mask_dir, image)
            mask_paths.append(mask_path)

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation1
        self.normalize_1 = normalize_1
        self.normalize_l4 = normalize_l4
        self.normalize_l3 = normalize_l3
        self.normalize_l2 = normalize_l2
        self.normalize_l1 = normalize_l1
        self.kwargs = kwargs

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = np.array(img)

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path)
        mask = np.array(mask)

        augment_1 = self.augmentation_1(image=img, mask=mask)
        img = augment_1['image']
        mask = augment_1['mask']

        img_ = np.array(Image.fromarray(img).convert('L'))
        _, l4, l3, l2, l1 = pywt.wavedec2(img_, 'db2', level=4)

        l4 = np.array(l4).transpose(1, 2, 0)
        l3 = np.array(l3).transpose(1, 2, 0)
        l2 = np.array(l2).transpose(1, 2, 0)
        l1 = np.array(l1).transpose(1, 2, 0)
        normalize_l4 = self.normalize_l4(image=l4)
        l4 = normalize_l4['image'].float()
        normalize_l3 = self.normalize_l3(image=l3)
        l3 = normalize_l3['image'].float()
        normalize_l2 = self.normalize_l2(image=l2)
        l2 = normalize_l2['image'].float()
        normalize_l1 = self.normalize_l1(image=l1)
        l1 = normalize_l1['image'].float()

        normalize_1 = self.normalize_1(image=img, mask=mask)
        img = normalize_1['image']
        mask = normalize_1['mask'].long()

        sampel = {'image': img, 'image_l1': l1, 'image_l2': l2, 'image_l3': l3, 'image_l4': l4, 'mask': mask, 'ID': os.path.split(mask_path)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths)


def imagefloder_aerial_lanenet(data_dir, data_transform, data_normalize, data_normalize_l1, data_normalize_l2, data_normalize_l3, data_normalize_l4, **kwargs):
    dataset = dataset_aerial_lanenet(data_dir=data_dir,
                           augmentation1=data_transform,
                           normalize_1=data_normalize,
                           normalize_l1=data_normalize_l1,
                           normalize_l2=data_normalize_l2,
                           normalize_l3=data_normalize_l3,
                           normalize_l4=data_normalize_l4,
                           **kwargs
                           )
    return dataset