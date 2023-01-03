import os
import sys
import glob
import random

import h5py
import numpy as np

import torch


CATEGORIES = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table']


class dataset_modelnet10(torch.utils.data.Dataset):
    def __init__(self, args, split, transforms):
        super().__init__()

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES

        pointers = glob.glob(os.path.join(args.path, args.dataset_type[:-2], '*', self.split, '*.npy'))

        self.shapes = list()
        self.labels = list()
        for pointer in pointers:
            self.shapes.append(pointer)
            self.labels.append(CATEGORIES.index(pointer.split('/')[-3]))

        # CAUTION, FOR PAPER VISUALIZATIONS ONLY
        # self.shapes = self.shapes[1:]
        # self.labels = self.labels[1:]

        self.n_instances = list()
        for label in np.unique(self.labels):
            partition = np.where(np.asarray(self.labels) == label)[0]
            self.n_instances.append(len(partition))

    def __getitem__(self, index):

        # Get label
        label = self.labels[index]
        assert label < self.args.n_classes, 'Warning: found label > n_classes'

        # Get vertices
        xyz = np.load(self.shapes[index])

        # Sampling
        idxs = np.random.choice(
            range(xyz.shape[0]),
            size=self.args.n_verts,
            replace=False if self.args.n_verts <= xyz.shape[0] else True)
        xyz = xyz[idxs, :]  # (n_verts, 3)

        if self.args.dataset_type == 'shapenet10' and label != CATEGORIES.index('plant'):
            # TODO Swap y and z coordinates
            # TODO xyz = xyz[:, [0, 2, 1]]
            # Rotate shapes so that the up axis is the z axis
            angle = -np.pi / 2
            R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            xyz = xyz.dot(R_x)

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


class dataset_shapenet10(dataset_modelnet10):
    pass


class dataset_scannet10(dataset_modelnet10):
    def __init__(self, args, split, transforms):
        super().__init__(args, split, transforms)

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES

        pointers = np.loadtxt(os.path.join(args.path, args.dataset_type[:-2], '{:s}_files.txt'.format(self.split)), dtype='str', ndmin=1)

        shapes_list = list()
        labels_list = list()
        for pointer in pointers:
            f = h5py.File(os.path.join(args.path, args.dataset_type[:-2], pointer), mode='r')
            shapes = f['data'][:]
            labels = f['label'][:]
            shapes_list.append(shapes)
            labels_list.append(labels)

        self.shapes = np.concatenate(shapes_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0).squeeze()

        self.n_instances = list()
        for label in np.unique(self.labels):
            partition = np.where(np.asarray(self.labels) == label)[0]
            self.n_instances.append(len(partition))

    def __getitem__(self, index):

        # Get label
        label = self.labels[index]
        assert label < self.args.n_classes, 'Warning: found label > n_classes'

        # Get vertices
        xyz = self.shapes[index, :, :3]

        # Sampling
        idxs = np.random.choice(
            range(xyz.shape[0]),
            size=self.args.n_verts,
            replace=False if self.args.n_verts <= xyz.shape[0] else True)
        xyz = xyz[idxs, :]  # (n_verts, 3)

        # TODO Swap y and z coordinates
        # TODO xyz = xyz[:, [0, 2, 1]]
        # Rotate shapes so that the up axis is the z axis
        angle = -np.pi / 2
        R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        xyz = xyz.dot(R_x)

        # Apply data transforms
        if self.transforms is not None:
            xyz = self.transforms(xyz)

        # Transpose vertices
        xyz = np.transpose(xyz)  # (3, n_verts)

        # Create PyTorch tensors
        xyz = torch.tensor(xyz, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return xyz, label
