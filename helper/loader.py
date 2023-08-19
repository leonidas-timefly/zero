import os

import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms


def data_loader(dataset, batch_size, train_set_path, test_set_path, num_workers=2, shuffle=True):

    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        print('==> Preparing CIFAR10 data..')
        train_set = torchvision.datasets.CIFAR10(root=train_set_path, train=True, download=True, transform=transforms_train)
        test_set = torchvision.datasets.CIFAR10(root=test_set_path, train=False, download=True, transform=transforms_test)
        class_number = 10
    elif dataset == 'stl10':
        print('==> Preparing STL10 data..')
        train_set = torchvision.datasets.STL10(root=train_set_path, split='train', download=True, transform=transforms_train)
        test_set = torchvision.datasets.STL10(root=test_set_path, split='test', download=True, transform=transforms_test)
        class_number = 10
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, class_number