import os
import h5py
import numpy as np
import torch
from utils import compute_voxelization


CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

CATEGORIES_COMMON = [
    ['dresser', 'wardrobe'], ['bench', 'chair', 'stool'], 'desk', 'monitor',
    'door', 'bookshelf', 'table', 'bed', 'sink', 'sofa', 'toilet']


class dataset_mn40(torch.utils.data.Dataset):
    '''
    ModelNet40 (MN40) classification dataset
    '''
    def __init__(self, args, split, transforms):
        super().__init__()

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES

        # Load data and labels
        data_train, label_train, data_test, label_test = self.load_data(self.args.path)
        if self.split == 'train':
            self.data = data_train
            self.label = label_train
        elif self.split == 'test':
            self.data = data_test
            self.label = label_test

        # Compute the number of instances for each class
        self.n_instances = list()
        for label in np.unique(self.label):
            partition = np.where(np.asarray(self.label) == label)[0]
            self.n_instances.append(len(partition))

    def load_h5(self, path):
        f = h5py.File(path, mode='r')
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def load_data(self, path):
        # Load training data
        data_train0, label_train0 = self.load_h5(os.path.join(path, 'ply_data_train0.h5'))
        data_train1, label_train1 = self.load_h5(os.path.join(path, 'ply_data_train1.h5'))
        data_train2, label_train2 = self.load_h5(os.path.join(path, 'ply_data_train2.h5'))
        data_train3, label_train3 = self.load_h5(os.path.join(path, 'ply_data_train3.h5'))
        data_train4, label_train4 = self.load_h5(os.path.join(path, 'ply_data_train4.h5'))
        # Load test data
        data_test0, label_test0 = self.load_h5(os.path.join(path, 'ply_data_test0.h5'))
        data_test1, label_test1 = self.load_h5(os.path.join(path, 'ply_data_test1.h5'))
        # Get training set
        data_train = np.concatenate( \
            [data_train0, data_train1, data_train2, data_train3, data_train4])
        label_train = np.concatenate( \
            [label_train0, label_train1, label_train2, label_train3, label_train4]).squeeze()
        # Get test set
        data_test = np.concatenate([data_test0, data_test1])
        label_test = np.concatenate([label_test0, label_test1]).squeeze()
        return data_train, label_train, data_test, label_test

    def __getitem__(self, index):

        # Get vertices
        xyz = self.data[index]  # (2048, 3)

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

        if self.args.flag_voxel is True:
            # Compute voxelization
            xyz = compute_voxelization(xyz, self.args)  # (n_voxels, ch, n_samples)
        else:
            # Transpose vertices
            xyz = np.transpose(xyz)  # (3, n_verts)

        # Get label
        label = self.label[index]

        # Create PyTorch tensors
        xyz = torch.tensor(xyz, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return xyz, label

    def __len__(self):
        return len(self.data)


class dataset_mn4011(dataset_mn40):
    def __init__(self, args, split, transforms):

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES_COMMON

        # Load data and labels
        data_train, label_train, data_test, label_test = \
            dataset_mn40(args, split, transforms).load_data(self.args.path)
        if self.split == 'train':
            data = data_train
            label = label_train
        elif self.split == 'test':
            data = data_test
            label = label_test

        # Select only common classes
        self.data = list()
        self.label = list()
        for id_common, category_common in enumerate(CATEGORIES_COMMON):

            if type(category_common) is list:
                for _ in range(len(category_common)):
                    id = CATEGORIES.index(category_common[_])
                    idxs = np.where(label == id)
                    self.data.append(data[idxs, :].squeeze(0))
                    self.label.append(id_common * np.ones_like(label[idxs]))

            else:
                id = CATEGORIES.index(category_common)
                idxs = np.where(label == id)
                self.data.append(data[idxs, :].squeeze(0))
                self.label.append(id_common * np.ones_like(label[idxs]))

        self.data = np.concatenate(self.data, axis=0)
        self.label = np.concatenate(self.label, axis=0)

        # Compute the number of instances for each class
        self.n_instances = list()
        for label in np.unique(self.label):
            partition = np.where(np.asarray(self.label) == label)[0]
            self.n_instances.append(len(partition))
