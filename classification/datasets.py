import os
from datasets_mn40 import dataset_mn40, dataset_mn4011
from datasets_sonn import dataset_sonn, dataset_sonn11
from datasets_pointda import dataset_modelnet10, dataset_shapenet10, dataset_scannet10
from datasets_grasp import dataset_rk10, dataset_rr10, dataset_sk10, dataset_sr10


def get_dataset(args, split, transforms):

    assert os.path.isdir(args.path), 'Error: The dataset path provided does not exist'

    # MN40
    if args.dataset_type == 'mn40':
        dataset = dataset_mn40(args, split, transforms)

    # SONN
    elif args.dataset_type == 'sonn':
        dataset = dataset_sonn(args, split, transforms)

    # MN40-SONN
    elif args.dataset_type == 'mn4011':
        dataset = dataset_mn4011(args, split, transforms)
    elif args.dataset_type == 'sonn11':
        dataset = dataset_sonn11(args, split, transforms)

    # PointDA
    elif args.dataset_type == 'modelnet10':
        dataset = dataset_modelnet10(args, split, transforms)
    elif args.dataset_type == 'shapenet10':
        dataset = dataset_shapenet10(args, split, transforms)
    elif args.dataset_type == 'scannet10':
        dataset = dataset_scannet10(args, split, transforms)

    # GraspNetPC
    elif args.dataset_type == 'rk10':
        dataset = dataset_rk10(args, split, transforms)
    elif args.dataset_type == 'rr10':
        dataset = dataset_rr10(args, split, transforms)
    elif args.dataset_type == 'sk10':
        dataset = dataset_sk10(args, split, transforms)
    elif args.dataset_type == 'sr10':
        dataset = dataset_sr10(args, split, transforms)

    return dataset
