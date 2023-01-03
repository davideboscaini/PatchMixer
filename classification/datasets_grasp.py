import os
import sys
import glob
import random

import h5py
import numpy as np

import torch


CATEGORIES = ['box', 'can', 'banana', 'powerdrill', 'scissors', 'pear', 'dish', 'camel', 'mouse', 'shampoo']


class dataset_rk10(torch.utils.data.Dataset):
    def __init__(self, args, split, transforms):
        super().__init__()
        pointers = glob.glob(os.path.join(args.path, split, 'Real', 'kinect', '*', '*.xyz'))
        self.init(args, split, transforms, pointers)

    def init(self, args, split, transforms, pointers):
        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES

        self.shapes = list()
        self.labels = list()
        for pointer in pointers:
            self.shapes.append(pointer)
            self.labels.append(int(pointer.split('/')[-2]))

        self.n_instances = list()
        for label in np.unique(self.labels):
            partition = np.where(np.asarray(self.labels) == label)[0]
            self.n_instances.append(len(partition))

    def load_xyz(self, pointer):
        f = open(pointer)
        xyz = list()
        for line in f:
            x, y, z = line.split()
            xyz.append([float(x), float(y), float(z)])
        f.close()
        return np.array(xyz)

    def __getitem__(self, index):

        # Get label
        label = self.labels[index]
        assert label < self.args.n_classes, 'Warning: found label > n_classes'

        # Get vertices
        xyz = self.load_xyz(self.shapes[index])

        # Sampling
        idxs = np.random.choice(
            range(xyz.shape[0]),
            size=self.args.n_verts,
            replace=False if self.args.n_verts <= xyz.shape[0] else True)
        xyz = xyz[idxs, :]  # (n_verts, 3)

        # Apply data transforms
        if self.transforms is not None:
            xyz = self.transforms(xyz)

        # Transpose vertices
        xyz = np.transpose(xyz)  # (3, n_verts)

        # Create PyTorch tensors
        xyz = torch.tensor(xyz, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return xyz, label

    def __len__(self):
        return len(self.shapes)


class dataset_rr10(dataset_rk10):
    def __init__(self, args, split, transforms):
        pointers = glob.glob(os.path.join(args.path, split, 'Real', 'realsense', '*', '*.xyz'))
        dataset_rk10.init(self, args, split, transforms, pointers)


class dataset_sk10(dataset_rk10):
    def __init__(self, args, split, transforms):
        pointers = glob.glob(os.path.join(args.path, split, 'Synthetic', 'kinect', '*', '*.xyz'))
        dataset_rk10.init(self, args, split, transforms, pointers)


class dataset_sr10(dataset_rk10):
    def __init__(self, args, split, transforms):
        pointers = glob.glob(os.path.join(args.path, split, 'Synthetic', 'realsense', '*', '*.xyz'))
        dataset_rk10.init(self, args, split, transforms, pointers)
