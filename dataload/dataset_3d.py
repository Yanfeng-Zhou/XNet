import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import torchio as tio
import SimpleITK as sitk
from torchio.data import UniformSampler, LabelSampler


class dataset_it(Dataset):
    def __init__(self, data_dir, input1, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_it, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        if sup:
            mask_dir = data_dir + '/mask'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            if sup:
                mask_path = os.path.join(mask_dir, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), mask=tio.LabelMap(mask_path), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )


class dataset_it_dtc(Dataset):
    def __init__(self, data_dir, input1, num_classes, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_it_dtc, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        if sup:
            mask_dir_1 = data_dir + '/mask'
            mask_dir_2 = data_dir + '/mask_sdf1'
            if num_classes == 3:
                mask_dir_3 = data_dir + '/mask_sdf2'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                mask_path_2 = os.path.join(mask_dir_2, i)
                if num_classes == 3:
                    mask_path_3 = os.path.join(mask_dir_3, i)
                    subject_1 = tio.Subject(
                        image=tio.ScalarImage(image_path_1),
                        mask=tio.LabelMap(mask_path_1),
                        mask2=tio.LabelMap(mask_path_2),
                        mask3=tio.LabelMap(mask_path_3),
                        ID=i)
                else:
                    subject_1 = tio.Subject(
                        image=tio.ScalarImage(image_path_1),
                        mask=tio.LabelMap(mask_path_1),
                        mask2=tio.LabelMap(mask_path_2),
                        ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

class dataset_iit(Dataset):
    def __init__(self, data_dir, input1, input2, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_iit, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        image_dir_2 = data_dir + '/' + input2

        if sup:
            mask_dir_1 = data_dir + '/mask'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            image_path_2 = os.path.join(image_dir_2, i)
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), mask=tio.LabelMap(mask_path_1), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )


class dataset_iit_conresnet(Dataset):
    def __init__(self, data_dir, input1, input2, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_iit_conresnet, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        image_dir_2 = data_dir + '/' + input2

        if sup:
            mask_dir_1 = data_dir + '/mask'
            mask_dir_2 = data_dir + '/mask_res'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            image_path_2 = os.path.join(image_dir_2, i)
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                mask_path_2 = os.path.join(mask_dir_2, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), mask=tio.LabelMap(mask_path_1), mask2=tio.LabelMap(mask_path_2), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

