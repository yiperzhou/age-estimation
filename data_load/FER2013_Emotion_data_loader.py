''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from torchvision import transforms

from .CVPR_16_ChaLearn_data_loader import dataset_augmentation_sampler

class FER2013(data.Dataset):
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File("/media/yi/harddrive/codes/MultitaskLearningFace/resources/FER2013/data.h5", 'r', driver='core')

        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))

        elif self.split == "PrivateTest":
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))
        else:
            print("FER2013 dataset, split should be ['Training', 'PublicTest', 'PrivateTest']")
            raise ValueError("split should be ['Training', 'PublicTest', 'PrivateTest']")


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':

            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
            


def get_FER2013_Emotion_data(args):

    emotion_transform = {
        "emotion_transform_train": transforms.Compose([
            # transforms.RandomCrop(44),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        "emotion_transform_valid": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        "emotion_transform_test": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    }

    # train data size: 28710
    emotion_trainset = FER2013(split='Training', transform=emotion_transform["emotion_transform_train"])
    emotion_train_loader = torch.utils.data.DataLoader(emotion_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    # emotion_train_loader = torch.utils.data.DataLoader(emotion_trainset, batch_size=args.batch_size, shuffle=False, 
    #                                                     num_workers=args.loading_jobs, sampler=dataset_augmentation_sampler(emotion_trainset, 4548))

    # valid data size: 3590
    emotion_validset = FER2013(split='PublicTest', transform=emotion_transform["emotion_transform_valid"])
    emotion_valid_loader = torch.utils.data.DataLoader(emotion_validset, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)

    # # test data size: 3590
    emotion_testset = FER2013(split='PrivateTest', transform=emotion_transform["emotion_transform_test"])
    emotion_test_loader = torch.utils.data.DataLoader(emotion_testset, batch_size=args.batch_size, shuffle=True, num_workers=args.loading_jobs)
    # emotion_test_loader = torch.utils.data.DataLoader(emotion_testset, batch_size=args.batch_size, shuffle=False, 
    #                                                         num_workers=args.loading_jobs, sampler=dataset_augmentation_sampler(emotion_testset, 3590))
    
    print('[FER2013 dataset] load: finished !')


    return [emotion_train_loader, emotion_test_loader]
