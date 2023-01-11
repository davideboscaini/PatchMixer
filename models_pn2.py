import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO Source is: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
# TODO If we change normalization technique we should also change the pointnet2 radius,
# default values should be validated for unit-ball normalization


def square_distance(src, dst):
    '''
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # TODO Sanity check negative distances
    # print(torch.min(dist))
    # assert torch.min(dist) >= 0, "square_distance fail: negative dist "
    return dist


def index_points(points, idx):
    '''
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    '''
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    '''
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    '''
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    '''
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    '''
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    '''
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        '''
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class SetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(SetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        '''
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        '''
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        '''
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class pn2_class_ssg(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Feature extraction
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # Classification head
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.p_drop),  # TODO 0.4
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.p_drop),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, args.n_classes, bias=True)
        )

    def forward(self, x):

        verts = x[:, :3, :]
        feats = x[:, 3:, :]

        if feats.nelement() == 0:
            feats = None

        # Set Abstraction layers
        l1_xyz, l1_feats = self.sa1(verts, feats)  # (bs, 3, 512), (bs, 128, 512)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)  # (bs, 3, 128), (bs, 256, 128)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)  # (bs, 3, 1), (bs, dim, 1)
        x = l3_feats.view(verts.shape[0], -1)  # (bs, dim)

        # Classification head
        x = self.fc1(x)
        y = self.fc2(x)
        x = self.fc3(y)

        return x, y


class pn2_class_msg(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Feature extraction
        self.sa1 = SetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = SetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128], in_channel=320, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640 + 3, mlp=[256, 512, 1024], group_all=True)

        # Classification head
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.p_drop),  # TODO 0.4
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.p_drop),  # TODO 0.4
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, args.n_classes, bias=True)
        )

    def forward(self, x):

        verts = x[:, :3, :]
        feats = x[:, 3:, :]

        if feats.nelement() == 0:
            feats = None

        # Set Abstraction layers
        l1_xyz, l1_feats = self.sa1(verts, feats)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats)
        x = l3_feats.view(verts.shape[0], -1)

        # Classification head
        x = self.fc1(x)
        y = self.fc2(x)
        x = self.fc3(y)

        return x, y













class pn2_feat_ssg(nn.Module):
    def __init__(self, dim_in=3):
        super(pn2_feat_ssg, self).__init__()
        self.dim_in = dim_in

        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=64,in_channel=self.dim_in, mlp=[64, 64, 128], group_all=False)  # Here yanx27 is using nsample=32, but original pn2 tf repo uses 64!
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = FeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=384, mlp=[256, 128])
        # TODO We do not pass orginal points coordinates to last FP layer
        # self.fp1 = FeaturePropagation(in_channel=128 + self.dim_in, mlp=[128, 128, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp=[128, 128, 128])

    def forward(self, verts, feats):
        bs = verts.shape[0]
        l0_xyz = verts # (bs, 3, n_verts=1024)
        l0_feats = feats # None
        # Set Abstraction layers
        l1_xyz, l1_feats = self.sa1(l0_xyz, l0_feats) # (bs, 3, 512), (bs, 128, 512)
        l2_xyz, l2_feats = self.sa2(l1_xyz, l1_feats) # (bs, 3, 128), (bs, 256, 128)
        l3_xyz, l3_feats = self.sa3(l2_xyz, l2_feats) # (bs, 3, 1), (bs, dim, 1)
        # Feature Propagation layers
        l2_feats = self.fp3(l2_xyz, l3_xyz, l2_feats, l3_feats)
        l1_feats = self.fp2(l1_xyz, l2_xyz, l1_feats, l2_feats)
        l0_feats = self.fp1(l0_xyz, l1_xyz, l0_feats, l1_feats)
        feat_class = l3_feats.view(bs, -1) # (bs, dim)
        feat_segm = l0_feats # (bs, 128, n_verts=1024)
        return feat_class, feat_segm


class pn2_segm(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Feature extraction
        self.feat = pn2_feat_ssg(dim_in=args.ch)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(args.p_drop),  # TODO 0.5
            nn.Conv1d(128, args.n_classes, 1)
        )

    def forward(self, x):

        verts = x[:, :3, :]
        feats = x[:, 3:, :]

        if feats.nelement() == 0:
            feats = None

        # Feature extraction
        _, feat_segm = self.feat(verts, feats)

        # Segmentation head
        x = self.seg_head(feat_segm)  # (bs, n_classes, n_verts)

        return x
