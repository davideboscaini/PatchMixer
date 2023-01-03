import numpy as np
from scipy.spatial.transform import Rotation as rotation
import torch
from torchvision.transforms import functional as F


class to_tensor(object):
    def __call__(self, xyz):
        xyz = torch.tensor(xyz, dtype=torch.float32)
        return xyz


class center(object):
    def __call__(self, xyz):
        xyz[:, :3] -= np.mean(xyz[:, :3], axis=0)
        return xyz


class scale1(object):
    '''
    PyTorch Geometric repo, normalize_scale in antoalli's repo
    '''
    def __call__(self, xyz):
        diag = np.max(np.absolute(xyz[:, :3]))
        xyz[:, :3] /= (diag + np.finfo(float).eps)
        return xyz


class scale2(object):
    '''
    fxia22 GitHub repo, center_normalize in antoalli's repo
    '''
    def __call__(self, xyz):
        diag = np.max(np.sqrt(np.sum(xyz[:, :3] ** 2, axis=1)))
        xyz[:, :3] /= (diag + np.finfo(float).eps)
        return xyz


class jitter1(object):
    '''
    jitter_pointcloud in antoalli's repo
    '''
    def __init__(self, sigma, clip):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, xyz):
        # Random samples from N(mu, sigma**2), i.e. sigma * np.random.randn() + mu
        jitter = self.sigma * np.random.randn(xyz.shape[0], 3)
        jitter = np.clip(jitter, -1.0 * self.clip, self.clip)
        xyz[:, :3] += jitter
        return xyz


class jitter2(object):
    '''
    From fxia22 GitHub repo
    '''
    def __init__(self, sigma, clip):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, xyz):
        jitter = np.random.normal(0, self.sigma, size=(xyz.shape[0], 3))
        jitter = np.clip(jitter, -1.0 * self.clip, self.clip)
        xyz[:, :3] += jitter
        return xyz


class rotate1(object):
    def __init__(self, max_ang_x, max_ang_y, max_ang_z):
        ang_x = np.random.uniform(-max_ang_x, max_ang_x)
        ang_y = np.random.uniform(-max_ang_y, max_ang_y)
        ang_z = np.random.uniform(-max_ang_z, max_ang_z)
        self.R = rotation.from_euler('zyx', [ang_z, ang_y, ang_x], degrees=True).as_matrix()

    def __call__(self, xyz):
        xyz[:, :3] = np.dot(xyz[:, :3], self.R)  # TODO Investigate
        return xyz


class rotate2(object):
    '''
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    '''
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, xyz):
        angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        if self.axis == 'x':
            R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
            xyz = np.matmul(xyz, R_x)
        elif self.axis == 'y':
            R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
            xyz = np.matmul(xyz, R_y)
        elif self.axis == 'z':
            R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
            xyz = np.matmul(xyz, R_z)
        return xyz


class normalize(object):
    def __call__(self, xyz):
        # xyz[:, 3:] /= np.linalg.norm(xyz[:, 3:], axis=0)
        xyz[:, 3:] /= np.max(np.abs(xyz[:, 3:]), axis=0)
        return xyz


class scale_and_translate(object):
    '''
    Source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/data.py
    '''
    def __init__(self, min_scale=2./3, max_scale=3./2, min_trans=-0.2, max_trans=0.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_trans = min_trans
        self.max_trans = max_trans

    def __call__(self, xyz):
        scale = np.random.uniform(low=self.min_scale, high=self.max_scale, size=[3])
        trans = np.random.uniform(low=self.min_trans, high=self.max_trans, size=[3])
        xyz = np.add(np.multiply(xyz, scale), trans).astype('float32')
        return xyz
