import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=torch.device('cuda')).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size * num_points, num_dims)  # batch_size * num_points * k + range(0, batch_size * num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class dgcnn_class(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * 3, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * 64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(2 * 128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * args.dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.p_drop)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.p_drop)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, args.n_classes, bias=True)
        )

    def forward(self, x):

        bs = x.size(0)
        x = get_graph_feature(x, k=self.args.k)  # (bs, 6, n_verts, k)
        x = self.conv1(x)  # (bs, 64, n_verts, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (bs, 64, n_verts)

        x = get_graph_feature(x1, k=self.args.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.args.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.args.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (bs, 64 + 64 + 128 + 256, n_verts)

        x = self.conv5(x)  # (bs, dim, n_verts)

        x1 = F.adaptive_max_pool1d(x, 1).view(bs, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(bs, -1)
        x = torch.cat((x1, x2), 1)  # (bs, 2 * dim)

        x = self.fc1(x)
        y = self.fc2(x)
        x = self.fc3(y)

        return x, y


class dgcnn_segm(nn.Module):
    '''
    Code from: https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py
    '''
    def __init__(self, args):
        super().__init__()

        self.args = args
        # TODO self.transform_net = Transform_Net(args)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, args.dim, kernel_size=1, bias=False),  # 192 = 64 + 64 + 64
            nn.BatchNorm1d(args.dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(1216, 256, kernel_size=1, bias=False),  # TODO 1280
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=args.p_drop)
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp2 = nn.Dropout(p=args.p_drop)
        self.conv10 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv11 = nn.Conv1d(128, args.n_classes, kernel_size=1, bias=False)

    def forward(self, x):  # TODO x, l
        batch_size = x.size(0)
        num_points = x.size(2)

        # TODO x0 = get_graph_feature(x, k=self.args.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # TODO t = self.transform_net(x0)                   # (batch_size, 3, 3)
        # TODO x = x.transpose(2, 1)                        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # TODO x = torch.bmm(x, t)                          # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # TODO x = x.transpose(2, 1)                        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.args.k)        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                              # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.args.k)       # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.args.k)       # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)             # (batch_size, 64*3, num_points)

        x = self.conv6(x)                              # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]             # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # TODO l = l.view(batch_size, -1, 1)                  # (batch_size, num_categories, 1)
        # TODO l = self.conv7(l)                              # (batch_size, num_categories, 1) -> (batch_size, 64, 1)

        # TODO x = torch.cat((x, l), dim=1)                   # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)                 # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)          # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                              # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)

        x = self.conv9(x)                              # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)

        x = self.conv10(x)                             # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                             # (batch_size, 256, num_points) -> (batch_size, n_classes, num_points)

        return x
