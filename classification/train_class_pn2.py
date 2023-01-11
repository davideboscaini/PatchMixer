import sys
sys.path.append('..')

import argparse
import os
import shutil
import random
import warnings
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torchvision
import torchinfo
from torch.utils.tensorboard import SummaryWriter

from augmentations import center, scale1, scale2, jitter1, jitter2, rotate1, rotate2, scale_and_translate
from datasets import get_dataset
from models_pn2 import pn2_class_ssg, pn2_class_msg
from losses import loss_ce, loss_cbce, loss_lsce
from utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='Computational device')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--flag_resume', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='Fix random seed')

    # Train
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Manual epoch number (useful on restarts)')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--freq_train', type=int, default=1, help='Frequence at which log training')
    parser.add_argument('--freq_test', type=int, default=None, help='Frequence at which log validation')
    parser.add_argument('--freq_save', type=int, default=None, help='Frequence at which save the model')

    # Data
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset_type', type=str, required=True, help='Which dataset to use')
    parser.add_argument('--n_verts', type=int, default=2048, help='Number of input samples')
    parser.add_argument('--flag_voxel', default=False, action='store_true', help='Use voxelization for the patch embedding')
    parser.add_argument('--variant', type=str, default=None)

    # Augmentations
    parser.add_argument('--augms', type=str, default=None, help='Sequence of data augmentation techniques to use (separated by commas)')
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--clip', type=float, default=0.05)
    parser.add_argument('--min_scale', type=float, default=0.67)
    parser.add_argument('--max_scale', type=float, default=1.5)
    parser.add_argument('--min_trans', type=float, default=-0.2)
    parser.add_argument('--max_trans', type=float, default=0.2)

    # Model
    parser.add_argument('--aggregation_type', type=str, required=True)
    parser.add_argument('--ch', type=int, default=3, help='Input dimension')
    parser.add_argument('--dim', type=int, default=1024, help='Feature dimension')
    parser.add_argument('--n_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--p_drop', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--activation_type', type=str, default='relu', help='Activation type')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint')

    # Optimizer
    parser.add_argument('--optim_type', type=str, default='SGD', help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-03, help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='Scheduler type')
    parser.add_argument('--step', type=int, default=None, help='Step for the lr decay')
    parser.add_argument('--gamma', type=float, default=1e-01, help='Multiplicative factor of lr decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument('--w_decay', type=float, default=1e-05, help='Weight decay')

    # Loss
    parser.add_argument('--loss_type', type=str, default='ce', help='Loss type')
    parser.add_argument('--beta', type=float, default=0.9, help='Class-balanced cross-entropy hyperparam')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Label smoothing hyperparam')

    args = parser.parse_args()
    return args


def main():

    # Parse input arguments
    args = parse_args()

    # Update weights and runs paths
    args.path_weights = os.path.join('..', 'data', 'exps', 'weights', args.exp)
    args.path_runs = os.path.join('..', 'data', 'exps', 'runs', args.exp)

    # Create experiment folder
    os.makedirs(args.path_weights, exist_ok=True)

    # Save the current model as backup
    os.system('cp ../models_pn2.py {:s}/models_pn2_backup.py'.format(args.path_weights))
    os.system('" " > {:s}/__init__.py'.format(args.path_weights))

    # Create logger
    logger = get_logger(os.path.join(args.path_weights, 'log_train.txt'))

    # Create TensorBoard logger
    writer = SummaryWriter(log_dir=args.path_runs)

    # Log library versions
    logger.info('PyTorch version = {:s}'.format(torch.__version__))
    logger.info('TorchVision version = {:s}'.format(torchvision.__version__))

    # Activate CUDNN backend
    torch.backends.cudnn.enabled = True

    # Fix random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    # Log input arguments
    for arg, value in vars(args).items():
        logger.info('{} = {}'.format(arg, value))

    # Perform the training
    run_train(args, logger, writer)


def run_train(args, logger, writer):

    # Get the data augmentations
    transforms_train_list = list()
    transforms_test_list = list()
    if args.augms is not None:
        augms_list = args.augms.split(',')
        if 'center' in augms_list:
            transforms_train_list.append(center())
            transforms_test_list.append(center())
        if 'scale1' in augms_list:
            transforms_train_list.append(scale1())
            transforms_test_list.append(scale1())
        if 'jitter1' in augms_list:
            transforms_train_list.append(jitter1(sigma=args.sigma, clip=args.clip))
        if 'rotate1' in augms_list:
            transforms_train_list.append(rotate1(args.max_ang_x, args.max_ang_y, args.max_ang_z))
        if 'rotate2' in augms_list:
            transforms_train_list.append(rotate2(axis='z'))
        if 'scale_and_translate' in augms_list:
            transforms_train_list.append(scale_and_translate(args.min_scale, args.max_scale, args.min_trans, args.max_trans))
    transforms_train = torchvision.transforms.Compose(transforms_train_list)
    transforms_test = torchvision.transforms.Compose(transforms_test_list)

    # Get the training dataset
    dataset_train = get_dataset(args, split='train', transforms=transforms_train)

    # Get the test dataset
    dataset_test = get_dataset(args, split='test', transforms=transforms_test)

    # Log train/test stats
    logger.info('TRN samples: {:d}, {}'.format(len(dataset_train), dataset_train.n_instances))
    logger.info('TST samples: {:d}, {}'.format(len(dataset_test), dataset_test.n_instances))

    # Get the training data loader
    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.bs,
        num_workers=args.n_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    # Get the test data loader
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        num_workers=args.n_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    # Get the model
    if args.aggregation_type == 'ssg':
        model = pn2_class_ssg(args)
    elif args.aggregation_type == 'msg':
        model = pn2_class_msg(args)

    # Send the model to the device
    model = model.to(args.device)

    # Set data parallelism
    if torch.cuda.device_count() == 1:
        logger.info('Using a single GPU, this will disable data parallelism')
    else:
        logger.info('Using multiple GPUs, with data parallelism')
        model = torch.nn.DataParallel(model)

    # Set the model in training mode
    model.train()

    # Visualize the learnable parameters and log them
    logger.info('Learnable parameters:')
    if hasattr(model, 'module'):
        params_to_update = model.module.parameters()
        for name, param in model.module.named_parameters():
            if param.requires_grad is True:
                logger.info(name)
    else:
        params_to_update = model.parameters()
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                logger.info(name)

    # Get the model summary
    if torch.cuda.device_count() == 1:
        logger.info('Model summary:')
        stats = torchinfo.summary(model, (args.bs, 3, args.n_verts))
        logger.info(str(stats))

    # Get the optimizer
    if args.optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            params=params_to_update,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.w_decay,
            nesterov=False)
    elif args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(
            params=params_to_update,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-04)
    else:
        raise NotImplementedError

    # Get the learning rate scheduler
    if args.step is None:
        args.step = args.n_epochs
    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            args.step - 1,
            args.gamma)
    elif args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.step - 1,
            eta_min=args.gamma * args.lr)
    elif args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=50)
    else:
        raise NotImplementedError

    # Get the classification loss
    if args.loss_type == 'ce':
        criterion1 = loss_ce()
    elif args.loss_type == 'cbce':
        criterion1 = loss_cbce(
            beta=args.beta,
            n_instances=dataset_train.n_instances)  # Warning, here we are using the training instances
    elif args.loss_type == 'lsce':
        criterion1 = loss_lsce(
            epsilon=args.epsilon,
            reduction='sum')
    else:
        raise NotImplementedError
    criterion1 = criterion1.to(args.device)

    # Init training stats
    oa_best = 0.0
    mca_best = 0.0
    epoch_oa_train_best = 0.0
    epoch_oa_test_best = 0.0
    epoch_ca_mean_train_best = 0.0
    epoch_ca_mean_test_best = 0.0
    since_train = time.time()

    # Loop over epochs
    for epoch in range(args.start_epoch + 1, args.n_epochs + 1):

        # Init epoch stats
        running_loss_train = 0.0
        running_oa_train = 0
        running_ca_pred_train = torch.zeros(args.n_classes).to(args.device)
        running_ca_target_train = torch.zeros(args.n_classes).to(args.device)
        running_grad_mean = 0.0
        running_grad_max = 0.0
        since_epoch_train = time.time()

        # Init confusion matrix
        confusion_matrix = torch.zeros(args.n_classes, args.n_classes, dtype=torch.int, device=args.device)

        # Iterate over training data
        for idx_batch, data_batch in enumerate(loader_train):

            # Load a mini-batch
            input, target = data_batch

            # Send data to device
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            # Train one iter
            loss, pred = train_one_iter(input, target, model, criterion1, optimizer, args)

            # Compute confusion matrix
            for i, j in zip(target, pred):
                confusion_matrix[i, j] += 1

            # Iteration statistics
            running_loss_train += loss.item() * input.shape[0]
            running_oa_train += torch.sum(pred == target.squeeze())
            running_ca_pred_train += torch.sum(
                torch.nn.functional.one_hot(pred, num_classes=args.n_classes) * \
                torch.nn.functional.one_hot(target, num_classes=args.n_classes), dim=0)
            running_ca_target_train += torch.sum(
                torch.nn.functional.one_hot(target, num_classes=args.n_classes), dim=0)

            # Log to tensorboard
            global_step = epoch * len(loader_train) + idx_batch
            # writer.add_scalar('Train-Iter/Loss', loss.item(), global_step)
            # writer.add_scalar('Train-Iter/Learning rate', optimizer.param_groups[0]['lr'], global_step)

            # Log model gradients to tensorboard
            grad_mean = 0.0
            grad_max = 0.0
            num_layers = 0
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # writer.add_scalar('Debug/model_grad_{}'.format(n), torch.mean(p.grad), global_step=global_step)
                    grad_mean += torch.mean(p.grad).item()
                    grad_max += torch.max(p.grad).item()
                    num_layers += 1
            grad_mean /= num_layers
            grad_max /= num_layers
            running_grad_mean += grad_mean
            running_grad_max += grad_max
            # writer.add_scalar('Train-Iter/Grad. mean', running_grad_mean, global_step)
            # writer.add_scalar('Train-Iter/Grad. max', running_grad_max, global_step)

        # Epoch statistics
        epoch_loss_train = running_loss_train / len(loader_train.dataset)
        epoch_oa_train = running_oa_train.float() / len(loader_train.dataset)
        epoch_ca_train = torch.div(running_ca_pred_train, running_ca_target_train)
        epoch_ca_mean_train = torch.mean(epoch_ca_train)
        epoch_ca_std_train = torch.std(epoch_ca_train)
        epoch_grad_mean = running_grad_mean / len(loader_train)
        epoch_grad_max = running_grad_max / len(loader_train)

        # Check best accuracy
        if epoch_oa_train > epoch_oa_train_best:
            epoch_oa_train_best = epoch_oa_train
        if epoch_ca_mean_train > epoch_ca_mean_train_best:
            epoch_ca_mean_train_best = epoch_ca_mean_train

        # Log to tensorboard
        writer.add_scalar('Train/Loss', epoch_loss_train, epoch)
        writer.add_scalar('Train/* Overall Acc.', epoch_oa_train, epoch)
        writer.add_scalar('Train/* Overall Acc. best', epoch_oa_train_best, epoch)
        writer.add_scalar('Train/Class Acc. mean', epoch_ca_mean_train, epoch)
        writer.add_scalar('Train/Class Acc. std', epoch_ca_std_train, epoch)
        writer.add_scalar('Train/Class Acc. mean best', epoch_ca_mean_train_best, epoch)
        writer.add_scalar('Train/Grad. mean', epoch_grad_mean, epoch)
        writer.add_scalar('Train/Grad. max', epoch_grad_max, epoch)
        writer.add_scalar('Train/Learning rate', optimizer.param_groups[0]['lr'], epoch)

        # Log training stats
        elapsed_epoch_train = time.time() - since_epoch_train
        if epoch % args.freq_train == 0:
            logger.info('TRN, Epoch: {:4d}, Loss: {:e}, OA: {:.4f}, MCA: {:.4f} +- {:.4f}, Elapsed: {:.1f}s'.format(
                epoch, epoch_loss_train, epoch_oa_train, epoch_ca_mean_train, epoch_ca_std_train, elapsed_epoch_train))

        # Plot confusion matrix
        if epoch % 10 == 0:
            plt.figure()
            cm = confusion_matrix.cpu().detach().numpy().astype('float32')
            for _ in range(cm.shape[0]):
                cm[_, :] /= ((1.0 * loader_train.dataset.n_instances[_]) + np.finfo(float).eps)
            ax = sns.heatmap(cm,
                cmap='Blues', cbar=False, vmin=0.0, vmax=1.0,
                annot=True, annot_kws={'fontsize': 4}, fmt='.2f',
                xticklabels=loader_train.dataset.categories, yticklabels=loader_train.dataset.categories)
            ax.tick_params(left=False, bottom=False)
            ax.set_xlabel('Predicted classes')
            ax.set_ylabel('Ground-truth classes')
            ax.set_title('Confusion matrix (oa = {:.4f}, ca = {:.4f} +- {:.4f})'.format(
                epoch_oa_train, epoch_ca_mean_train, epoch_ca_std_train))
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(4)
            os.makedirs(os.path.join(args.path_weights, 'cm'), exist_ok=True)
            plt.savefig(
                os.path.join(args.path_weights, 'cm', 'cm_train_epoch={:04d}.png'.format(epoch)),
                transparent=False, bbox_inches='tight', dpi=300)
            plt.close()

        # Iterate over test data
        if args.freq_test is not None:
            if epoch % args.freq_test == 0:

                # Init test
                running_loss_test = 0.0
                running_oa_test = 0
                running_ca_pred_test = torch.zeros(args.n_classes).to(args.device)
                running_ca_target_test = torch.zeros(args.n_classes).to(args.device)
                since_epoch_test = time.time()

                # Init confusion matrix
                confusion_matrix = torch.zeros(args.n_classes, args.n_classes, dtype=torch.int, device=args.device)

                # Loop over test data
                for idx_batch, data_batch in enumerate(loader_test):

                    # Load a mini-batch
                    input, target = data_batch

                    # Send data to device
                    input = input.to(args.device, non_blocking=True)
                    target = target.to(args.device, non_blocking=True)

                    # Test one iter
                    loss, pred = test_one_iter(input, target, model, criterion1, args)

                    # Compute confusion matrix
                    for i, j in zip(target, pred):
                        confusion_matrix[i, j] += 1

                    # Iteration statistics
                    running_loss_test += loss.item() * input.shape[0]
                    running_oa_test += torch.sum(pred == target.squeeze())
                    running_ca_pred_test += torch.sum(
                        torch.nn.functional.one_hot(pred, num_classes=args.n_classes) * \
                        torch.nn.functional.one_hot(target, num_classes=args.n_classes), dim=0)
                    running_ca_target_test += torch.sum(
                        torch.nn.functional.one_hot(target, num_classes=args.n_classes), dim=0)

                # Epoch statistics
                epoch_loss_test = running_loss_test / len(loader_test.dataset)
                epoch_oa_test = running_oa_test.float() / len(loader_test.dataset)
                epoch_ca_test = torch.div(running_ca_pred_test, running_ca_target_test)
                epoch_ca_mean_test = torch.mean(epoch_ca_test)
                epoch_ca_std_test = torch.std(epoch_ca_test)

                # Check best accuracy
                if epoch_oa_test > epoch_oa_test_best:
                    epoch_oa_test_best = epoch_oa_test
                if epoch_ca_mean_test > epoch_ca_mean_test_best:
                    epoch_ca_mean_test_best = epoch_ca_mean_test

                # Log to tensorboard
                writer.add_scalar('Test/Loss', epoch_loss_test, epoch)
                writer.add_scalar('Test/* Overall Acc.', epoch_oa_test, epoch)
                writer.add_scalar('Test/* Overall Acc. best', epoch_oa_test_best, epoch)
                writer.add_scalar('Test/Class Acc. mean', epoch_ca_mean_test, epoch)
                writer.add_scalar('Test/Class Acc. std', epoch_ca_std_test, epoch)
                writer.add_scalar('Test/Class Acc. mean best', epoch_ca_mean_test_best, epoch)

                # Log test stats
                elapsed_epoch_test = time.time() - since_epoch_test
                logger.info('TST, Epoch: {:4d}, Loss: {:e}, OA: {:.4f}, MCA: {:.4f} +- {:.4f}, Elapsed: {:.1f}s'.format(
                    epoch, epoch_loss_test, epoch_oa_test, epoch_ca_mean_test, epoch_ca_std_test, elapsed_epoch_test))

                # Plot confusion matrix
                if epoch % 10 == 0:
                    plt.figure()
                    cm = confusion_matrix.cpu().detach().numpy().astype('float32')
                    for _ in range(cm.shape[0]):
                        cm[_, :] /= 1.0 * loader_test.dataset.n_instances[_] + np.finfo(float).eps
                    ax = sns.heatmap(cm,
                        cmap='Blues', cbar=False, vmin=0.0, vmax=1.0,
                        annot=True, annot_kws={'fontsize': 4}, fmt='.2f',
                        xticklabels=loader_test.dataset.categories, yticklabels=loader_test.dataset.categories)  # mask=cm==0.0
                    ax.tick_params(left=False, bottom=False)
                    ax.set_xlabel('Predicted classes')
                    ax.set_ylabel('Ground-truth classes')
                    ax.set_title('Confusion matrix (oa = {:.4f}, ca = {:.4f} +- {:.4f})'.format(
                        epoch_oa_test, epoch_ca_mean_test, epoch_ca_std_test))
                    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(4)
                    os.makedirs(os.path.join(args.path_weights, 'cm'), exist_ok=True)
                    plt.savefig(
                        os.path.join(args.path_weights, 'cm', 'cm_test_epoch={:04d}.png'.format(epoch)),
                        transparent=False, bbox_inches='tight', dpi=300)
                    plt.close()

                # Save the best model
                if epoch_oa_test >= oa_best:
                    oa_best = epoch_oa_test
                    torch.save({
                        'args': args,
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'sched_state_dict': scheduler.state_dict(),
                    },
                    os.path.join(args.path_weights, 'best_oa.pth'))
                # TODO if epoch_ca_mean_test >= mca_best:
                #     mca_best = epoch_ca_mean_test
                #     torch.save({
                #         'args': args,
                #         'epoch': epoch,
                #         'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                #         'optim_state_dict': optimizer.state_dict(),
                #         'sched_state_dict': scheduler.state_dict(),
                #     },
                #     os.path.join(args.path_weights, 'best_mca.pth'))

        # TODO Save the current model
        # if epoch % args.freq_save == 0:
        # 'epoch={:04d}.pth'.format(epoch)

        # Save the last model
        if epoch == args.n_epochs:
            torch.save({
                'args': args,
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                },
                os.path.join(args.path_weights, 'last.pth'))

        # Scheduler step
        if args.scheduler_type == 'plateau':
            scheduler.step(epoch_loss_test)
        else:
            scheduler.step()

    # Log timining
    elapsed_train = time.time() - since_train
    logger.info('Training completed in {:.2f}s'.format(elapsed_train))


def train_one_iter(input, target, model, criterion1, optimizer, args):

    # Set the model in training mode
    model.train()

    # Zero the parameters gradients
    optimizer.zero_grad()

    # Forward pass
    logits, _ = model(input)
    _, pred = torch.max(logits, dim=1)

    # Classification loss
    loss = criterion1(logits, target)

    # Back-propagation
    loss.backward()

    # Optimizer step
    optimizer.step()

    return loss, pred


def test_one_iter(input, target, model, criterion1, args):

    # Set the model in evaluation mode
    model.eval()

    # Forward pass
    with torch.inference_mode():
        logits, _ = model(input)
        _, pred = torch.max(logits, dim=1)

    # Classification loss
    loss = criterion1(logits, target)

    return loss, pred


if __name__ == '__main__':
    main()
