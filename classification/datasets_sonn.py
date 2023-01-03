import os
import h5py
import numpy as np
import torch
from utils import compute_voxelization


CATEGORIES = [
    'bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door',
    'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']

CATEGORIES_COMMON = [
    'cabinet', 'chair', 'desk', 'display', 'door',
    'shelf', 'table', 'bed', 'sink', 'sofa', 'toilet']


class dataset_sonn(torch.utils.data.Dataset):
    '''
    ScanObjectNN (SONN) classification dataset
    '''
    def __init__(self, args, split, transforms):
        super().__init__()

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES

        # Load data and labels
        if self.split == 'train':
            if self.args.variant == 'nobg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'training_objectdataset.h5'), 'r')
            elif self.args.variant == 'bg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'training_objectdataset.h5'), 'r')
            elif self.args.variant == 'nobg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            elif self.args.variant == 'bg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.data = np.array(h5['data']).astype(np.float32)
            self.label = np.array(h5['label']).astype(int)
            h5.close()

        elif self.split == 'test':
            if self.args.variant == 'nobg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'test_objectdataset.h5'), 'r')
            if self.args.variant == 'bg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'test_objectdataset.h5'), 'r')
            elif self.args.variant == 'nobg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            elif self.args.variant == 'bg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.data = np.array(h5['data']).astype(np.float32)
            self.label = np.array(h5['label']).astype(int)
            h5.close()

        # Compute the number of instances for each class
        self.n_instances = list()
        for label in np.unique(self.label):
            partition = np.where(np.asarray(self.label) == label)[0]
            self.n_instances.append(len(partition))

    def __getitem__(self, index):

        # Get vertices
        xyz = self.data[index]

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


class dataset_sonn11(dataset_sonn):
    def __init__(self, args, split, transforms):

        self.args = args
        self.split = split
        self.transforms = transforms
        self.categories = CATEGORIES_COMMON

        if self.split == 'train':
            if self.args.variant == 'nobg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'training_objectdataset.h5'), 'r')
            elif self.args.variant == 'bg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'training_objectdataset.h5'), 'r')
            elif self.args.variant == 'nobg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            elif self.args.variant == 'bg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            data = np.array(h5['data']).astype(np.float32)  # (2309, 2048, 3)
            label = np.array(h5['label']).astype(int)  # (2309, )
            h5.close()

        elif self.split == 'test':
            if self.args.variant == 'nobg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'test_objectdataset.h5'), 'r')
            if self.args.variant == 'bg_noaugm':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'test_objectdataset.h5'), 'r')
            elif self.args.variant == 'nobg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split_nobg', 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            elif self.args.variant == 'bg_pbt50rs':
                h5 = h5py.File(os.path.join(args.path, 'main_split', 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            data = np.array(h5['data']).astype(np.float32)  # (1701, 2048, 3)
            label = np.array(h5['label']).astype(int)  # (1701, )
            h5.close()

        # Select only common classes
        self.data = list()
        self.label = list()
        for id_common, category_common in enumerate(CATEGORIES_COMMON):
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
