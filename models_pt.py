import os
import sys
import torch
import torch.nn as nn

sys.path.insert(1, os.path.abspath('../point-transformer/scene_seg'))
from model.point_transformer.point_transformer_seg import PointTransformerCla


class dgcnn_class(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = PointTransformerCla(c=3, k=args.n_classes, args=args)

    def forward(self, x):
        x = self.model(x)
        return x
