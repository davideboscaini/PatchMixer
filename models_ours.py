from einops.layers.torch import Reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_pn2 import farthest_point_sample, index_points, query_ball_point, knn_point, sample_and_group
from models_dgcnn import knn, get_graph_feature
from utils import viz_shape, viz_shape_parts

# Conv > Normalization > Activation > Dropout > Pooling (from https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout)


def init_activation(args):
    if args.activation_type.lower() == 'relu':
        return nn.ReLU(inplace=True)
    elif args.activation_type.lower() == 'leaky':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)  # TODO negative_slope=0.2, 0.01
    elif args.activation_type.lower() == 'gelu':
        return nn.GELU()
    elif args.activation_type.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif args.activation_type.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif args.activation_type.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif args.activation_type.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)


def shake(weight, gamma):
    with torch.no_grad():
        # weigths_std = torch.std(weight).item()
        weights_to_sum = torch.zeros_like(weight)
        torch.nn.init.xavier_uniform_(weights_to_sum)
        weights_to_sum *= gamma
        weight += weights_to_sum
    return weight


class mlp1(nn.Module):
    def __init__(self, dim_in, dim_out, args):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.activation = init_activation(args)

        self.lin1 = nn.Sequential(
            # TODO nn.Dropout(0.1),
            nn.Conv1d(self.dim_in, self.dim_out, kernel_size=1, bias=True, groups=1),
            nn.BatchNorm1d(self.dim_out),
        )

        if args.shake:
            self.lin1[0].weight = shake(self.lin1[0].weight, args.shake_factor)

        # TODO Weight and bias normalization
        # torch.nn.init.xavier_uniform_(self.lin1[0].weight, gain=nn.init.calculate_gain('relu'))
        # torch.nn.init.zeros_(self.lin1[0].bias)

        self.lin2 = nn.Sequential(
            # TODO nn.Dropout(0.1),
            nn.Conv1d(self.dim_out, self.dim_in, kernel_size=1, bias=True, groups=1),
            nn.BatchNorm1d(self.dim_in),
        )

        if args.shake:
            self.lin2[0].weight = shake(self.lin2[0].weight, args.shake_factor)

    def forward(self, x):
        x = self.lin1(x)  # (B, D', P)
        x = self.activation(x)
        x = self.lin2(x)  # (B, D, P)
        return x


class mlp2(nn.Module):
    def __init__(self, dim_in, dim_out, args):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.args = args

        self.activation = init_activation(args)

        self.lin1 = nn.Sequential(
            nn.Conv2d(2 * self.dim_in, self.dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.dim_out),
        )

        self.lin2 = nn.Sequential(
            nn.Conv2d(2 * self.dim_out, self.dim_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.dim_in),
        )

    def forward(self, x):

        y = get_graph_feature(x, k=self.args.k)  # (bs, 2 * n_patches, dim, k)
        y = self.lin1(y)  # (bs, 512, dim, k)
        x = y.max(dim=-1, keepdim=False)[0]  # (bs, 512, dim)

        x = self.activation(x)

        y = get_graph_feature(x, k=self.args.k)  # (bs, 2 * 512, dim, k)
        y = self.lin2(y)  # (bs, n_patches, dim, k)
        x = y.max(dim=-1, keepdim=False)[0]  # (bs, n_patches, dim)

        return x


class get_idxs_fps(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):

        # Number of patches to extract
        n_patches = int(self.args.patch_erasing * self.args.n_patches)

        # Shape vertices
        x0 = x.transpose(1, 2).contiguous()  # (bs, n_verts, 3)

        # Centroids absolute coordinates
        idxs_fps = farthest_point_sample(x0, n_patches)  # (bs, n_patches)
        x1 = index_points(x0, idxs_fps)  # (bs, n_patches, 3)

        return x0, x1, idxs_fps


class get_patches(nn.Module):
    def __init__(self, radius, args):
        super().__init__()
        self.radius = radius
        self.args = args

    def forward(self, x0, x1):

        bs = x0.shape[0]
        n_patches = x1.shape[1]

        if self.args.query_type == 'ball':
            idxs_patches = query_ball_point(self.radius, self.args.n_samples, x0, x1)
        elif self.args.query_type == 'knn':
            idxs_patches = knn_point(self.args.n_samples, x0, x1)
        elif self.args.query_type == 'both':
            idxs_ball = query_ball_point(self.radius, self.args.n_samples // 2, x0, x1)  # (B, PM * P, S / 2)
            idxs_knn = knn_point(self.args.n_samples // 2, x0, x1)  # (B, PM * P, S / 2)
            idxs_patches = torch.cat((idxs_ball, idxs_knn), dim=-1)  # (B, PM * P, S)

        # Absolute coordinates
        x2 = index_points(x0, idxs_patches)  # (B, PM * P, S, 3)

        # Old version (bugged):
        # x2 = x2 - x1.unsqueeze(2)
        # dists = torch.cdist(
        #     x2.view(bs * n_patches, self.args.n_samples, 3).contiguous(),
        #     x1.view(bs * n_patches, 3).contiguous().unsqueeze(1),
        #     p=2.0).view(bs, n_patches, self.args.n_samples, 1).contiguous()  # (bs, n_patches, n_samples, 1)
        # x3 = torch.cat((x2, dists), dim=3)

        # Distance from centroid
        if self.args.concat_dists:
            dists = torch.cdist(
                x2.view(bs * n_patches, self.args.n_samples, 3).contiguous(),
                x1.view(bs * n_patches, 3).contiguous().unsqueeze(1),
                p=2.0).view(bs, n_patches, self.args.n_samples, 1).contiguous()  # (B, P, S, 1)
            # values, idxs = torch.max(dists, dim=2)

        # Relative coordinates
        if self.args.coord_type == 'relative':
            if self.args.query_type == 'ball':
                x3 = x2 - x1.unsqueeze(2)
                # x3 = (x2 - x1.unsqueeze(2)) / self.radius
            elif self.args.query_type == 'knn':
                x3 = x2 - x1.unsqueeze(2)
                # x3 = torch.div(x2 - x1.unsqueeze(2), values.unsqueeze(-1))  # (B, 1.5P, S, 3), (B, 1.5P, 1, 1)
            elif self.args.query_type == 'both':
                x3 = x2 - x1.unsqueeze(2)

        if self.args.concat_dists:
            # if self.args.query_type == 'ball' or self.args.query_type == 'knn':
            #     dists = torch.div(dists, values.unsqueeze(-1))
            x3 = torch.cat((x3, dists), dim=3)

        # Concatenate with centroids
        x4 = x1.unsqueeze(2).repeat(1, 1, self.args.n_samples, 1)  # (B, P, S, 3)
        x5 = torch.cat((x3, x4), dim=3)

        out = x5.transpose(3, 1).transpose(2, 3).contiguous()  # (B, 3+3+1=7, P, S)

        with torch.no_grad():
            centroids = torch.clone(x1).transpose(1, 2).contiguous()

        return out, centroids, idxs_patches


class embedding_ball_pn(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.dim = dim
        self.args = args

        self.activation = init_activation(args)

        if self.args.concat_dists:
            self.dim_input = 7
        else:
            self.dim_input = 6

        if args.embedding_mlp_type == 'single':

            if args.mlp_type == 'mlp1':
                self.lin = nn.Sequential(
                    nn.Conv1d(self.dim_input, self.dim, kernel_size=1, bias=True),
                    nn.BatchNorm1d(self.dim),
                    self.activation)
                if args.shake:
                    self.lin[0].weight = shake(self.lin[0].weight, args.shake_factor)

            elif args.mlp_type == 'mlp2':
                self.lin = nn.Sequential(
                    nn.Conv2d(2 * self.dim_input, self.dim, kernel_size=1, bias=True),
                    nn.BatchNorm2d(self.dim),
                    self.activation)

        elif args.embedding_mlp_type == 'multi':

            if args.mlp_type == 'mlp1':
                self.lin1 = nn.Sequential(
                    nn.Conv1d(self.dim_input, 16, 1, bias=True),
                    nn.BatchNorm1d(16),
                    self.activation)
                if args.shake:
                    self.lin1[0].weight = shake(self.lin1[0].weight, args.shake_factor)

                self.lin2 = nn.Sequential(
                    nn.Conv1d(16, 32, 1, bias=True),
                    nn.BatchNorm1d(32),
                    self.activation)
                if args.shake:
                    self.lin2[0].weight = shake(self.lin2[0].weight, args.shake_factor)

                self.lin3 = nn.Sequential(
                    nn.Conv1d(32, 64, 1, bias=True),
                    nn.BatchNorm1d(64),
                    self.activation)
                if args.shake:
                    self.lin3[0].weight = shake(self.lin3[0].weight, args.shake_factor)

                self.lin4 = nn.Sequential(
                    nn.Conv1d(64, 128, 1, bias=True),
                    nn.BatchNorm1d(128),
                    self.activation)
                if args.shake:
                    self.lin4[0].weight = shake(self.lin4[0].weight, args.shake_factor)

                self.lin5 = nn.Sequential(
                    nn.Conv1d(128, self.dim, 1, bias=True),
                    nn.BatchNorm1d(self.dim),
                    self.activation)
                if args.shake:
                    self.lin5[0].weight = shake(self.lin5[0].weight, args.shake_factor)

            elif args.mlp_type == 'mlp2':
                self.lin1 = nn.Sequential(
                    nn.Conv2d(2 * self.dim_input, 16, kernel_size=1, bias=True),
                    nn.BatchNorm2d(16),
                    self.activation)
                self.lin2 = nn.Sequential(
                    nn.Conv2d(2 * 16, 32, kernel_size=1, bias=True),
                    nn.BatchNorm2d(32),
                    self.activation)
                self.lin3 = nn.Sequential(
                    nn.Conv2d(2 * 32, 64, kernel_size=1, bias=True),
                    nn.BatchNorm2d(64),
                    self.activation)
                self.lin4 = nn.Sequential(
                    nn.Conv2d(2 * 64, 128, kernel_size=1, bias=True),
                    nn.BatchNorm2d(128),
                    self.activation)
                self.lin5 = nn.Sequential(
                    nn.Conv2d(2 * 128, self.dim, kernel_size=1, bias=True),
                    nn.BatchNorm2d(self.dim),
                    self.activation)

    def forward(self, x):

        bs = x.shape[0]
        ch = x.shape[1]

        x = x.transpose(1, 2).contiguous()  # (bs, n_patches, ch, n_samples)
        x = x.view(bs * self.args.n_patches, ch, self.args.n_samples).contiguous()  # (bs * n_patches, ch, n_samples)

        if self.args.embedding_mlp_type == 'single':

            if self.args.mlp_type == 'mlp1':
                x = self.lin(x)

            elif self.args.mlp_type == 'mlp2':
                y = get_graph_feature(x, k=self.args.k)
                y = self.lin(y)
                x = y.max(dim=-1, keepdim=False)[0]

        elif self.args.embedding_mlp_type == 'multi':

            if self.args.mlp_type == 'mlp1':
                x = self.lin1(x)
                x = self.lin2(x)
                x = self.lin3(x)
                x = self.lin4(x)
                x = self.lin5(x)  # (bs * n_patches, dim, n_samples)

            elif self.args.mlp_type == 'mlp2':
                y = get_graph_feature(x, k=self.args.k)
                y = self.lin1(y)
                x = y.max(dim=-1, keepdim=False)[0]
            
                y = get_graph_feature(x, k=self.args.k)
                y = self.lin2(y)
                x = y.max(dim=-1, keepdim=False)[0]

                y = get_graph_feature(x, k=self.args.k)
                y = self.lin3(y)
                x = y.max(dim=-1, keepdim=False)[0]

                y = get_graph_feature(x, k=self.args.k)
                y = self.lin4(y)
                x = y.max(dim=-1, keepdim=False)[0]

                y = get_graph_feature(x, k=self.args.k)
                y = self.lin5(y)
                x = y.max(dim=-1, keepdim=False)[0]

        x = x.view(bs, self.args.n_patches, self.dim, self.args.n_samples).contiguous()  # (bs, n_patches, dim, n_samples)

        x = x.transpose(1, 2).contiguous()  # (bs, dim, n_patches, n_samples)

        return x


class token_mixer_layer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.activation = init_activation(args)

        self.norm = nn.LayerNorm(args.dim)

        if args.mlp_type == 'mlp1':
            self.mlp = mlp1(args.n_patches, args.hidden_dim_token, args)
        elif args.mlp_type == 'mlp2':
            self.mlp = mlp2(args.n_patches, args.hidden_dim_token, args)

        # Init first attention layer
        if args.attention_type == 1:
            self.attention1 = nn.Sequential(
                nn.Conv1d(args.dim, 1, 1, bias=True),
                nn.BatchNorm1d(1),
                nn.Sigmoid(),
            )
        elif args.attention_type == 2:
            self.attention1 = nn.Sequential(
                nn.Conv1d(args.dim, args.dim, 1, bias=True),
                nn.BatchNorm1d(args.dim),
                nn.Sigmoid(),
            )
        elif args.attention_type == 3 or args.attention_type == 4:
            self.attention1 = nn.Sequential(
                nn.Conv1d(args.n_patches, args.n_patches, 1, bias=True),
                nn.BatchNorm1d(args.n_patches),
                nn.Sigmoid(),
            )
        else:
            self.attention1 = None
        # self.attention1 = None

        # Init second attention layer
        if args.attention_type == 3 or args.attention_type == 4:
            self.attention2 = nn.Sequential(
                nn.Conv1d(args.hidden_dim_token, args.hidden_dim_token, 1, bias=True),
                nn.BatchNorm1d(args.hidden_dim_token),
                nn.Sigmoid(),
            )
        else:
            self.attention2 = None
        # self.attention2 = None

        # TODO self.attention = nn.Sequential(
        #     nn.Conv1d(args.dim, args.hidden_dim_token, 1, bias=False),
        #     nn.BatchNorm1d(args.hidden_dim_token),
        #     nn.Sigmoid(),
        # )

        # v1
        # self.attention = nn.Sequential(
        #     nn.Linear(args.dim, args.dim, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.0),
        #     nn.Linear(args.dim, args.hidden_dim_token, bias=True),
        #     nn.Dropout(0.0),
        # )

        # v2
        # self.attention = nn.Sequential(
        #     nn.Conv1d(args.n_patches, args.n_patches, 1, bias=False),
        #     nn.BatchNorm1d(args.n_patches),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.0),
        #     nn.Conv1d(args.n_patches, args.hidden_dim_token, 1, bias=True),
        #     nn.BatchNorm1d(args.hidden_dim_token),
        #     nn.Dropout(0.0)
        # )

        # self.dropout = nn.Dropout(0.0)

    def forward(self, x):

        x1 = x.transpose(2, 1).contiguous()  # (bs, n_patches, dim)
        x1 = self.norm(x1)  # (bs, n_patches, dim)

        if self.args.mlp_type == 'mlp1':

            # Apply first attention layer
            if self.args.attention_type == 1:
                y1 = self.attention1(x1.transpose(2, 1).contiguous())
                y1 = F.normalize(y1, p=2, dim=2)
                y1 = y1.transpose(2, 1)  # (bs, n_patches, 1)
                x2 = self.mlp.lin1(x1 * y1)
            elif self.args.attention_type == 2:
                y1 = self.attention1(x1.transpose(2, 1).contiguous())
                # y1 = F.normalize(y1, p=2, dim=2)
                y1 = y1.transpose(2, 1)  # (bs, n_patches, dim)
                x2 = self.mlp.lin1(x1 + y1)
            elif self.args.attention_type == 3:
                y1 = self.attention1(x1)  # (bs, n_patches, dim)
                # y1 = F.normalize(y1, p=2, dim=1)
                x2 = self.mlp.lin1(x1 + y1)
            elif self.args.attention_type == 4:
                y1 = self.attention1(x1)  # (bs, n_patches, dim)
                # y1 = F.normalize(y1, p=2, dim=1)
                x2 = self.mlp.lin1(torch.mul(x1, y1))
            elif self.args.attention_type == 5:
                feat = x1.transpose(2, 1).contiguous()  # (bs, dim, n_patches)
                idxs_knn = knn(feat, k=32)  # (bs, n_patches, k)
                y1 = torch.zeros(idxs_knn.shape[0], idxs_knn.shape[1], idxs_knn.shape[1]).to(x1.device)  # (bs, n_patches, n_patches)
                for idx_batch in range(idxs_knn.shape[0]):
                    for row in range(idxs_knn.shape[1]):
                        _idxs_knn = idxs_knn[idx_batch, row, :]
                        y1[idx_batch, row, _idxs_knn] = 1.0 * torch.ones_like(_idxs_knn).to(x1.device)
                x2 = self.mlp.lin1(torch.bmm(y1, x1))  # (bs, dim, exp_factor * n_patches)
            else:
                y1 = None
                x2 = self.mlp.lin1(x1)
        # y1 = None

            x3 = self.mlp.activation(x2)

            # Apply second attention layer
            if self.args.attention_type == 3:
                y2 = self.attention2(x3)
                x4 = self.mlp.lin2(x3 + y2)
            elif self.args.attention_type == 4:
                y2 = self.attention2(x3)
                x4 = self.mlp.lin2(torch.mul(x3, y2))
            else:
                y2 = None
                x4 = self.mlp.lin2(x3)
            # y2 = None
            # x4 = self.mlp.lin2(x3)
        # y2 = None

            x5 = x4.transpose(2, 1).contiguous()  # (bs, dim, n_patches)
            out = x5 + x  # (bs, dim, n_patches)

        elif self.args.mlp_type == 'mlp2':

            x2 = self.mlp(x1)
            x3 = x2.transpose(2, 1).contiguous()  # (bs, dim, n_patches)
            out = x3 + x  # (bs, dim, n_patches)
            y1 = None
            y2 = None

        # v1
        # _x = torch.clone(x)  # (B, D, P)
        # weights = self.attention(_x)  # (B, H, P)
        # _x = self.drop(self.activation(weights @ _x.transpose(1, 2)))  # (B, H, D)
        # _x = self.drop(weights.transpose(1, 2) @ _x)  # (B, P, D)
        # _x = _x.transpose(1, 2).contiguous()  # (B, D, P)
        # out = _x + x  # (B, D, P)

        # v2
        # _x = torch.clone(x)  # (B, D, P)
        # _x = _x.transpose(2, 1).contiguous()  # (B, P, D)
        # _x = self.norm(_x)  # (B, P, D)
        # weights = self.attention(_x)  # (B, H, D)
        # _x = self.dropout(self.activation(weights @ _x.transpose(1, 2)))  # (B, H, P)
        # _x = self.dropout(weights.transpose(1, 2) @ _x)  # (B, D, P)
        # out = _x + x  # (B, D, P)

        return out, y1, y2


class channel_mixer_layer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.activation = init_activation(args)

        self.norm = nn.LayerNorm(args.dim)

        if args.mlp_type == 'mlp1':
            self.mlp = mlp1(args.dim, args.hidden_dim_channel, args)
        elif args.mlp_type == 'mlp2':
            self.mlp = mlp2(args.dim, args.hidden_dim_channel, args)

    def forward(self, x):

        x = x.transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(2, 1).contiguous()

        x1 = self.mlp(x)

        # Skip connection
        out = x1 + x

        return out


class mixer_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_mixer_layer = token_mixer_layer(args)
        self.channel_mixer_layer = channel_mixer_layer(args)
        self.token_mixer_layer.__init__(args)
        self.channel_mixer_layer.__init__(args)

    def forward(self, x):
        x, y1, y2 = self.token_mixer_layer(x)
        # y1, y2 = None, None
        x = self.channel_mixer_layer(x)
        return x, y1, y2


class head(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.activation = init_activation(args)

        self.fc1 = nn.Sequential(
            nn.Linear(args.dim, 512, bias=True),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Dropout(p=args.p_drop))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Dropout(p=args.p_drop))
        self.fc3 = nn.Sequential(
            nn.Linear(256, args.n_classes, bias=True))

    def forward(self, x):
        x = self.fc1(x)
        x_feat = self.fc2(x)
        x = self.fc3(x_feat)
        return x, x_feat


class mixer_class_singleres(nn.Module):
    '''
    Classification model using only a single resolution
    '''
    def __init__(self, args):
        super().__init__()

        assert len(args.radii) == 1, 'More than one radius specified'

        self.args = args
        self.radius = args.radii[0]

        # Init FPS sampling
        self.idxs = get_idxs_fps(args)

        # Init patch extraction
        self.patches = get_patches(self.radius, args)

        # Init patch embedding
        self.embedding = embedding_ball_pn(args.dim, args)

        # Init first pooling
        self.pool1 = Reduce('b p d s -> b p d', 'max')

        # Init mixer layers (https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
        if self.args.share_mixer_weights:
            self.mixer_layer = mixer_layer(args)
        else:
            self.mixer_layers = nn.ModuleList([mixer_layer(args) for _ in range(args.depth)])

        # Init normalization
        self.norm = nn.LayerNorm(self.args.dim)

        # Init second pooling
        self.pool2 = Reduce('b c n -> b c', 'max')
        # F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)

        # Init head
        self.head = head(args)

    def forward(self, x):

        others = dict()

        # FPS
        xa, xb, idxs_fps = self.idxs(x)

        # Extract patches
        x, others['centroids'], idxs_ball = self.patches(xa, xb)  # (bs, 7, n_patches, n_samples)

        # TODO Shuffling
        # x = x[:, torch.randperm(7), :, :]
        # x = x[:, :, torch.randperm(self.args.n_patches), :]
        # x = x[:, :, :, torch.randperm(self.args.n_samples)]

        # Erase patches randomly
        x = x[:, :, :self.args.n_patches, :]
        # idxs_ball = idxs_ball[:, :self.args.n_patches, :]

        # Get patch coordinates
        if self.args.coord_type == 'relative':
            if self.args.concat_dists:
                others['patches'] = x[:, :3, :, :] + x[0, 4:, :, :]
            else:
                others['patches'] = x[:, :3, :, :] + x[0, 3:, :, :]
        elif self.args.coord_type == 'absolute':
            others['patches'] = x[:, :3, :, :]

        # Compute patch embeddings
        x = self.embedding(x)  # (bs, dim, n_patches, n_samples)

        # Pooling
        x = self.pool1(x)  # (bs, dim, n_patches)
        others['feat_embedding'] = x

        # Mixer layers
        if self.args.share_mixer_weights:
            for idx in range(self.args.depth):
                x, y1, y2 = self.mixer_layer(x)  # (bs, dim, n_patches)
                others['feat_layer{:d}'.format(idx)] = x
                others['attention1_layer{:d}'.format(idx)] = y1
                others['attention2_layer{:d}'.format(idx)] = y2
                # Normalization
                x = x.transpose(2, 1).contiguous()
                x = self.norm(x)
                x = x.transpose(2, 1).contiguous()  # (bs, dim, n_patches)
        else:
            for idx, layer in enumerate(self.mixer_layers):
                x, y1, y2 = layer(x)  # (bs, dim, n_patches)
                others['feat_layer{:d}'.format(idx)] = x
                others['attention1_layer{:d}'.format(idx)] = y1
                others['attention2_layer{:d}'.format(idx)] = y2
                # Normalization
                x = x.transpose(2, 1).contiguous()
                x = self.norm(x)
                x = x.transpose(2, 1).contiguous()  # (bs, dim, n_patches)

        # Pooling
        x = self.pool2(x)  # (bs, dim)

        # Classification head
        x, feat = self.head(x)  # (bs, n_classes)
        others['features'] = feat

        return x, others


class mixer_class_multires(nn.Module):
    '''
    Classification model using multiple resolutions
    '''
    def __init__(self, args):
        super().__init__()

        assert len(args.radii) > 1, 'Only a single radius specified'

        self.args = args

        self.activation = init_activation(args)

        # Init FPS sampling
        self.idxs = get_idxs_fps(args)

        # Init patch extraction
        self.patch_layers = nn.ModuleList(
            [get_patches(radius, args) for radius in args.radii]
        )

        # Init patch embedding
        self.emb_layers = nn.ModuleList(
            # [embedding_ball_pn(int(args.dim / len(args.radii)), args) for _ in range(len(args.radii))]
            [embedding_ball_pn(args.dim, args) for _ in range(len(args.radii))]
        )

        # Init first pooling
        self.pool1 = Reduce('b p d s -> b p d', 'max')

        # Dimensionality reduction
        self.conv1 = nn.Sequential(
            nn.Conv1d(len(args.radii) * args.dim, args.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.dim),
            self.activation,
        )

        # Init mixer layers (https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
        self.mixer_layers = nn.ModuleList(
            [mixer_layer(args) for _ in range(args.depth)]
        )

        # Init normalization
        self.norm = nn.LayerNorm(self.args.dim)

        # Init second pooling
        self.pool2 = Reduce('b c n -> b c', 'max')

        # Init head
        self.head = head(args)

    def forward(self, x):

        others = dict()

        x0 = torch.clone(x)

        # FPS
        xa, xb, idxs_fps = self.idxs(x0)

        x1_list = list()
        for layer in self.patch_layers:
            # Extract patches
            x, _, idxs_ball = layer(xa, xb)

            # Erase patches randomly
            x1_list.append(x[:, :, :self.args.n_patches, :])

        x2_list = list()
        for idx, layer in enumerate(self.emb_layers):
            # Compute patch embeddings
            x = layer(x1_list[idx])  # (bs, dim, n_patches, n_samples)

            # Pooling
            x2_list.append(self.pool1(x))  # (bs, dim, n_patches)

        # Concatenate patch embeddings
        x = torch.cat(x2_list, dim=1)  # (bs, n_radii * dim, n_patches)

        # Dimensionality reduction
        x = self.conv1(x)  # (bs, dim, n_patches)

        # Mixer layers
        for idx, layer in enumerate(self.mixer_layers):
            x, y1, y2 = layer(x)  # (bs, dim, n_patches)
            others['attention1_layer{:d}'.format(idx)] = y1
            others['attention2_layer{:d}'.format(idx)] = y2

        # Normalization
        x = x.transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(2, 1).contiguous()  # (bs, dim, n_patches)

        # Pooling
        x = self.pool2(x)  # (bs, dim)

        # Classification head
        x, feat = self.head(x)  # (bs, n_classes)

        return x, others


class mixer_segm_singleres(nn.Module):
    '''
    Part segmentation model using only a single resolution
    '''
    def __init__(self, args):
        super().__init__()

        assert len(args.radii) == 1, 'More than one radius specified'

        self.args = args
        self.radius = args.radii[0]

        # Init FPS sampling
        self.idxs = get_idxs_fps(args)

        # Init patch extraction
        self.patches = get_patches(self.radius, args)

        # Init patch embedding
        self.embedding = embedding_ball_pn(args.dim, args)

        # Init first pooling
        self.pool1 = Reduce('b p d s -> b p d', 'max')

        # Init mixer layers (https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
        if self.args.share_mixer_weights:
            self.mixer_layer = mixer_layer(args)
        else:
            self.mixer_layers = nn.ModuleList([mixer_layer(args) for _ in range(args.depth)])

        # Init normalization
        self.norm = nn.LayerNorm(self.args.dim)

        # Init second pooling
        self.pool2 = Reduce('b c n -> b c', 'max')

        # Init head
        self.activation = init_activation(args)
        self.fc0 = nn.Sequential(
            nn.Conv1d(3 * args.dim, args.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.dim),
            self.activation,
            nn.Dropout(p=args.p_drop))
        self.fc1 = nn.Sequential(
            nn.Conv1d(args.dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Dropout(p=args.p_drop))
        self.fc2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Dropout(p=args.p_drop))
        self.fc3 = nn.Sequential(
            nn.Conv1d(256, args.n_classes, kernel_size=1, bias=True))

    def forward(self, x, labels):

        bs = x.shape[0]

        # Compute FPS
        xa, xb, _ = self.idxs(x)  # (B, V, 3), (B, P, 3)

        # Extract patches
        x, _, idxs_ball = self.patches(xa, xb)  # x (B, 7, P', S), idxs_fps (B, P'), idxs_ball (B, P', S), where P' = args.patch_erasing * P

        # Erase patches randomly
        x = x[:, :, :self.args.n_patches, :]  # (B, 7, P, S)
        idxs_ball = idxs_ball[:, :self.args.n_patches, :]  # (B, P, S)

        # Compute patch embeddings
        x = self.embedding(x)  # (B, F, P, S)
        x0 = torch.clone(x)

        # Pooling
        x = self.pool1(x)  # (B, F, P)
        x1 = torch.clone(x)

        # Apply mixer layers
        if self.args.share_mixer_weights:
            for idx in range(self.args.depth):
                x, y1, y2 = self.mixer_layer(x)  # (B, F, P)
                # Normalization
                x = x.transpose(2, 1).contiguous()
                x = self.norm(x)
                x = x.transpose(2, 1).contiguous()  # (B, F, P)
        else:
            for idx, layer in enumerate(self.mixer_layers):
                x, y1, y2 = layer(x)  # (B, F, P)
                # Normalization
                x = x.transpose(2, 1).contiguous()
                x = self.norm(x)
                x = x.transpose(2, 1).contiguous()  # (B, F, P)

        # Pooling
        x = self.pool2(x)  # (B, F)

        # Repeat
        x = x.unsqueeze(-1).repeat(1, 1, self.args.n_patches)  # (B, F, P)

        # Concatenate
        x = torch.cat((x, x1), dim=1)  # (B, 2F, P)

        # Repeat
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.args.n_samples)  # (B, 2F, P, S)

        # Concatenate
        x = torch.cat((x, x0), dim=1)  # (B, 3F, P, S)

        # Reshape
        x = x.reshape(bs, x.shape[1], x.shape[2] * x.shape[3]).contiguous()  # (B, 3F, PS)

        # Part segmentation head
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (B, C, PS)

        # !!! Sampling target

        # Option 1
        # labels = torch.ones((bs, self.args.n_patches), dtype=torch.long).to(self.args.device)
        # for row in range(bs):
        #     labels[row, :] = _labels[row, idxs_fps[row, :]]
        # labels = torch.gather(_labels, 1, idxs_fps)  # (bs, n_patches)

        # Option 2
        idxs = idxs_ball.reshape(bs, -1)
        labels = torch.gather(labels, 1, idxs)  # (B, PS)

        # Option 3
        # idxs has shape (bs, n_patches * n_samples)
        #    x has shape (bs, n_parts, n_patches * n_samples)
        #  out has shape (bs, n_parts, n_verts)
        # idxs = idxs_ball.reshape(x.shape[0], -1)  # (bs, n_patches * n_samples)
        # _labels = torch.gather(labels, 1, idxs)  # (bs, n_patches * n_samples)
        # out = torch.ones((x.shape[0], self.args.n_classes, self.args.n_verts), requires_grad=True).to(self.args.device)  # (bs, n_classes, n_verts)
        # for i in range(x.shape[2]):  # Loop on m * s
        #     for j in range(x.shape[0]):  # Loop on b
        #         out[j, :, _labels[j, i]] = out[j, :, _labels[j, i]] * x[j, :, i]  # x[j, :, i]
        # out = out.scatter(2, idxs.unsqueeze(1).repeat(1, self.args.n_classes, 1), x)  # reduce='multiply'

        # !!! Prediciton reduction
        with torch.no_grad():

            # No reduce
            # _probs = torch.clone(x)
            # probs = torch.zeros((bs, self.args.n_classes, self.args.n_verts), requires_grad=True).to(self.args.device)  # (bs, n_classes, n_verts)
            # probs = probs.scatter(2, idxs.unsqueeze(1).repeat(1, self.args.n_classes, 1), _probs)

            # Add
            _probs = torch.clone(x)  # (B, C, PS)
            # _probs = torch.nn.functional.softmax(torch.clone(x), 1)
            probs = torch.zeros((bs, self.args.n_classes, self.args.n_verts), requires_grad=True).to(self.args.device)  # (B, C, V)
            probs = probs.scatter(2, idxs.unsqueeze(1).repeat(1, self.args.n_classes, 1), _probs, reduce='add')

            # Multiply
            # _probs = torch.clone(x)
            # # _probs = torch.nn.functional.softmax(torch.clone(x), 1)
            # probs = torch.ones((bs, self.args.n_classes, self.args.n_verts), requires_grad=True).to(self.args.device)  # (bs, n_classes, n_verts)
            # probs = probs.scatter(2, idxs.unsqueeze(1).repeat(1, self.args.n_classes, 1), _probs, reduce='multiply')

        return x, labels, probs


class mixer_segm_multires(nn.Module):
    '''
    Part segmentation model using multiple resolutions
    '''
    def __init__(self, args):
        super().__init__()

        assert len(args.radii) > 1, 'Only a single radius specified'

        self.args = args

        self.activation = init_activation(args)

        # Init FPS sampling
        self.idxs = get_idxs_fps(args)

        # Init patch extraction
        self.patch_layers = nn.ModuleList(
            [get_patches(radius, args) for radius in args.radii]
        )

        # Init patch embedding
        self.emb_layers = nn.ModuleList(
            [embedding_ball_pn(args.dim, args) for _ in range(len(args.radii))]
        )

        # Init first pooling
        self.pool1 = Reduce('b p d s -> b p d', 'max')

        # TODO
        self.conv1 = nn.Sequential(
            nn.Conv1d(len(args.radii) * args.dim, args.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.dim),  # TODO
            self.activation,
        )

        # Init mixer layers (https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
        self.mixer_layers = nn.ModuleList(
            [mixer_layer(args) for _ in range(args.depth)]
        )

        # Init normalization
        self.norm = nn.LayerNorm(args.dim)

        # Init second pooling
        self.pool2 = Reduce('b c n -> b c', 'max')

        # Init head
        self.fc0 = nn.Sequential(
            nn.Conv1d(3 * args.dim, args.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.dim),
            self.activation,
            nn.Dropout(p=args.p_drop)
        )
        self.fc1 = nn.Sequential(
            nn.Conv1d(args.dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            self.activation,
            nn.Dropout(p=args.p_drop)
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Dropout(p=args.p_drop),
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(256, args.n_classes, kernel_size=1, bias=True)
        )

    def forward(self, x, labels):

        bs = x.shape[0]

        x0 = torch.clone(x)  # (bs, 3, n_verts)

        # Compute FPS
        xa, xb, _ = self.idxs(x0)

        # Extract patches
        x1_list = list()
        i1_list = list()
        for layer in self.patch_layers:
            x, _, i = layer(xa, xb)  # x has shape (bs, 7, n_patches, n_samples), i has shape (bs, n_patches, n_samples)

            # Erase patches randomly
            x1_list.append(x[:, :, :self.args.n_patches, :])
            i1_list.append(i[:, :self.args.n_patches, :])

        # Compute patch embeddings
        x2_list = list()
        x3_list = list()
        for idx, layer in enumerate(self.emb_layers):
            x = layer(x1_list[idx])

            x2_list.append(x)  # (bs, dim, n_patches, n_samples)

            # Pooling
            x3_list.append(self.pool1(x))  # (bs, dim, n_patches)

        # Concatenate patch embeddings
        x = torch.cat(x3_list, dim=1)  # (bs, n_radii * dim, n_patches)

        # TODO
        x = self.conv1(x)  # (bs, dim, n_patches)  # OLD
        _x = torch.clone(x)

        # Mixer layers
        for idx, layer in enumerate(self.mixer_layers):
            x, y1, y2 = layer(x)  # (bs, dim, n_patches)

        # Normalization
        x = x.transpose(2, 1).contiguous()
        x = self.norm(x)
        x = x.transpose(2, 1).contiguous()  # (bs, dim, n_patches)

        # Pooling
        x = self.pool2(x)  # (bs, dim)

        # Repeat
        x = x.unsqueeze(-1).repeat(1, 1, self.args.n_patches)  # (bs, dim, n_patches)

        # Concatenate
        # x = torch.cat((x, torch.cat(x3_list, dim=1)), dim=1)  # (bs, 3 * dim, n_patches)  # OLD
        x = torch.cat((x, _x), dim=1)  # (bs, 2 * dim, n_patches)

        # Repeat
        x = x.unsqueeze(-1).repeat(1, 1, 1, len(self.args.radii) * self.args.n_samples)  # (bs, 2 * dim, n_patches, n_radii * n_samples)

        # Concatenate
        x = torch.cat((x, torch.cat(x2_list, dim=3)), dim=1)  # (bs, 3 * dim, n_patches, n_radii * n_samples)

        # TODO
        x = x.reshape(bs, 3 * self.args.dim, -1).contiguous()  # (bs, 3 * dim, n_patches * n_radii * n_samples)

        # Part segmentation head
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (bs, n_classes, n_patches * n_radii * n_samples)

        # TODO
        i = torch.cat(i1_list, dim=2)  # (bs, n_patches, n_radii * n_samples)
        idxs = i.reshape(bs, -1)
        labels = torch.gather(labels, 1, idxs)  # (bs, n_patches * n_radii * n_samples)

        # TODO
        with torch.no_grad():
            _probs = torch.clone(x)  # (bs, n_classes, n_patches * n_samples)
            probs = torch.zeros((bs, self.args.n_classes, self.args.n_verts), requires_grad=True).to(self.args.device)  # (bs, n_classes, n_verts)
            probs = probs.scatter(2, idxs.unsqueeze(1).repeat(1, self.args.n_classes, 1), _probs, reduce='add')

        return x, labels, probs
