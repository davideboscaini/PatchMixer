import re
import os
import logging
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt


def sorted_alphanumeric(data):
    '''
    https://gist.github.com/SeanSyue/8c8ff717681e9ecffc8e43a686e68fd9
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def get_logger(path_log):
    '''
    https://www.toptal.com/python/in-depth-python-logging
    '''

    # Get logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    # Get formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Get file handler and add it to logger
    fh = logging.FileHandler(path_log, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Get console handler and add it to logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger


def compute_voxelization(xyz, args):
    '''
    Perform the voxelization of a point cloud
    '''

    grid_size = int(np.cbrt(args.n_patches))
    n_samples = args.n_samples
    ch = args.ch

    df = pd.DataFrame(xyz)
    df.columns = ['x', 'y', 'z']
    pcd = PyntCloud(df)

    # If regular_bounding_box=True, use a bounding box with edges of equal lenghts.
    # Otherwise, use the bounding box of the shape.
    idxs_voxelgrid = pcd.add_structure(
        'voxelgrid',
        n_x=grid_size,
        n_y=grid_size,
        n_z=grid_size,
        regular_bounding_box=False)  # WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING

    voxelgrid = pcd.structures[idxs_voxelgrid]
    idxs_xyz_voxelgrid = voxelgrid.query(xyz)

    voxel_centers = voxelgrid.voxel_centers

    # xyz_out = np.zeros((0, ch))
    # label_out = np.zeros((0, 1))
    # n_voxels = grid_size ** 3
    # for idx_voxel in range(n_voxels):
    #     xyz_voxel = np.vstack((xyz[idxs_xyz_voxelgrid == idx_voxel], voxel_centers[idx_voxel, :]))
    #     xyz_out = np.vstack((xyz_out, xyz_voxel))
    #     label_out = np.vstack((label_out, idx_voxel * np.ones((len(xyz_voxel), 1))))

    n_voxels = grid_size ** 3
    parts_out = list()

    for idx_voxel in range(n_voxels):
        xyz_voxel = np.vstack((xyz[idxs_xyz_voxelgrid == idx_voxel], voxel_centers[idx_voxel, :]))
        xyz_center = np.repeat(np.expand_dims(voxel_centers[idx_voxel, :], axis=0), repeats=xyz_voxel.shape[0], axis=0)

        if args.coord_type == 'relative':
            # Relative coordinates
            tmp = xyz_voxel - xyz_center
        elif args.coord_type == 'absolute':
            # Absolute coordinates
            tmp = xyz_voxel

        # Concatenate 0/1
        if ch == 4:
            if tmp.shape[0] == 1:
                tmp = np.hstack((tmp, 0.0 * np.ones((tmp.shape[0], 1))))
            else:
                tmp = np.hstack((tmp, 1.0 * np.ones((tmp.shape[0], 1))))

        parts_out.append(tmp)

    # Sampling with replacement
    xyz_out = np.zeros((n_voxels, ch, n_samples))
    for idx in range(n_voxels):
        if len(parts_out[idx]) > n_samples:
            idxs = np.random.choice(range(len(parts_out[idx])), size=n_samples, replace=False)
        else:
            idxs = np.random.choice(range(len(parts_out[idx])), size=n_samples, replace=True)
        xyz_out[idx, :, :] = np.transpose(parts_out[idx][idxs, :])

    return xyz_out


def viz_shape(xyzs, labels, n_rows, name_output, args):

    for idx in range(n_rows):

        xyz = xyzs[idx, :, :]
        if labels is not None:
            label = labels[idx]

        fig = plt.figure(figsize=(5, 20))

        ax = fig.add_subplot(n_rows, 4, 4 * idx + 1, projection='3d')
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=1.0, alpha=0.25, c='blue', cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        if labels is not None:
            ax.set_title('Label = {:d}'.format(label))
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(4)
        for item in (ax.get_xaxis().get_major_ticks() + ax.get_yaxis().get_major_ticks() + ax.get_zaxis().get_major_ticks()):
            item.set_pad(-4.)
        plt.grid()
        # ax.view_init(elev=10.0, azim=20.0)

        ax = fig.add_subplot(n_rows, 4, 4 * idx + 2, projection='3d')
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=1.0, alpha=0.25, c='blue', cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(4)
        for item in (ax.get_xaxis().get_major_ticks() + ax.get_yaxis().get_major_ticks() + ax.get_zaxis().get_major_ticks()):
            item.set_pad(-4.)
        plt.grid()
        ax.view_init(elev=0.0, azim=0.0)

        ax = fig.add_subplot(n_rows, 4, 4 * idx + 3, projection='3d')
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=1.0, alpha=0.25, c='blue', cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(4)
        for item in (ax.get_xaxis().get_major_ticks() + ax.get_yaxis().get_major_ticks() + ax.get_zaxis().get_major_ticks()):
            item.set_pad(-4.)
        plt.grid()
        ax.view_init(elev=0.0, azim=90.0)

        ax = fig.add_subplot(n_rows, 4, 4 * idx + 4, projection='3d')
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=1.0, alpha=0.25, c='blue', cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(4)
        for item in (ax.get_xaxis().get_major_ticks() + ax.get_yaxis().get_major_ticks() + ax.get_zaxis().get_major_ticks()):
            item.set_pad(-4.)
        plt.grid()
        ax.view_init(elev=90.0, azim=0.0)

        fig.savefig(
            os.path.join('models', args.exp, '{:s}_{:02d}.png'.format(name_output, idx)),
            transparent=False, bbox_inches='tight', dpi=300)
        plt.close()


def viz_shape_parts(xyz_lists, bs, name_output, args):

    for idx in range(bs):

        xyz_list = xyz_lists[idx]

        fig = plt.figure(figsize=(15, 10))

        xyz_all = np.zeros((0, 3))
        label_all = np.zeros((0, 1))

        for idx_part in range(len(xyz_list)):

            xyz = xyz_list[idx_part]

            ax = fig.add_subplot(5, int(np.ceil(len(xyz_list) / 5)), idx_part + 2, projection='3d')
            ax.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                s=2.0, alpha=0.5, c='blue', cmap='Paired')
            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-1, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.grid()
            # ax.view_init(elev=10.0, azim=20.0)

            xyz_all = np.vstack((xyz_all, xyz))
            label_all = np.vstack((label_all, idx_part * np.ones((len(xyz), 1))))

        ax = fig.add_subplot(5, int(np.ceil(len(xyz_list) / 5)), 1, projection='3d')
        ax.scatter(
            xyz_all[:, 0], xyz_all[:, 1], xyz_all[:, 2],
            s=2.0, alpha=0.5, c=label_all, cmap='Paired')
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.grid()
        # ax.view_init(elev=10.0, azim=20.0)

        fig.savefig(
            os.path.join('models', args.exp, '{:s}_{:02d}.png'.format(name_output, idx)),
            transparent=False, bbox_inches='tight', dpi=300)
        plt.close()
