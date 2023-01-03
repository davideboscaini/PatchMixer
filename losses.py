import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_ce(nn.Module):
    '''
    Cross-entropy loss
    '''
    def __init__(self):
        super(loss_ce, self).__init__()
        self.criterion = nn.CrossEntropyLoss()  # reduction='none'

    def forward(self, input, target):
        # input = F.log_softmax(input, dim=1)
        # loss = F.nll_loss(input, target, reduction='none')
        loss = self.criterion(input, target)
        return loss


class loss_cbce(nn.Module):
    '''
    Class-balanced cross-entropy loss
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples, CVPR 2019
    '''
    def __init__(self, beta, n_instances):
        super(loss_cbce, self).__init__()
        weight = torch.Tensor([(1 - beta) / (1 - beta**n) for n in n_instances])
        # weight /= torch.sum(weight)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        return self.criterion(input, target)


class loss_lsce(nn.Module):
    '''
    Label smoothing cross-entropy loss
    Muller et al., When Does Label Smoothing Help?, NeurIPS 2019
    '''
    def __init__(self, epsilon, reduction):
        super(loss_lsce, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        # Old code:
        # n = input.size()[-1]
        # log_preds = F.log_softmax(input, dim=-1)
        # loss = self._reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        # nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # loss = self._linear_combination(loss / n, nll, self.epsilon)

        n_class = input.size(1)
        one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = F.log_softmax(input, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()

        return loss

    def _reduce_loss(self, loss, reduction):
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y


class loss_focal(nn.Module):
    '''
    Focal loss
    Lin et al., Focal Loss for Dense Object Detection, ICCV 2017
    '''
    def __init__(self, gamma, reduction):
        super(loss_focal, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logpt = self.loss_ce(input, target)
        pt = torch.exp(-1.0 * logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class loss_reg(nn.Module):
    '''
    Regularization loss, \lvert R R^\top - I \rvert
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input):
        iden = torch.eye(input.shape[2], device='cuda').repeat(input.shape[0], 1, 1)
        diff = torch.bmm(input, input.transpose(2, 1)) - iden
        loss = torch.mean(torch.norm(diff, dim=(1, 2)))
        return loss
