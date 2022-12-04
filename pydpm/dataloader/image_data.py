#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Xinyang Liu <lxy771258012@163.com>
# License: BSD-3-Clause

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as datasets

def tensor_transforms(data, transforms=lambda x:x):
    data = data.numpy()
    data = transforms(data)
    return data

# ======================================== CustomDataset ======================================================== #

class CustomDataset(Dataset):
    def __init__(self, file_path, mode='train', transform=None, target_transform=None):
        super(CustomDataset, self).__init__()
        self.file_path = os.path.join(file_path, mode)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = []
        self.classes_num = 0
        self.class_to_idx = {}
        self.image_names = []
        self.image_classes = []
        self.classes_file = os.path.join(file_path, 'label.txt')

        with open(self.classes_file, 'r') as classes_list:
            for line in classes_list:
                self.classes_num += 1
                self.classes.append(line)
                self.class_to_idx[line] = self.classes_num - 1

        with open(self.file_path, 'r') as image_class_file:
            for line in image_class_file:
                image_class_pair = line.split('\t')
                self.image_names.append(image_class_pair[0])
                self.image_classes.append(image_class_pair[1])

    def __getitem__(self, idx):
        image_path, target = self.image_names[idx], self.class_to_idx[self.image_classes[idx]]

        # Return a PIL Image
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_names)


def image_dataloader(root='../dataset/mnist', transform=None, target_transform=None,
               batch_size=500, shuffle=True, drop_last=True, num_workers=4):
    dataset = CustomDataset(root, transform=transform, target_transform=target_transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last), dataset.classes

