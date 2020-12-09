import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from data import TrainDataset
import os
path = os.getcwd()


def get_network():
    from torchvision import models
    resnet = models.resnet18(pretrained=True)
    device = torch.device("cuda")
    resnet = resnet.to(device)
    return resnet


"""
class WarmUpLR(optim.lr_scheduler):
    # warmup_training learning rate scheduler
    # Args:
    #     optimizer: optimzier(e.g. SGD)
    #     total_iters: totoal_iters of warmup phase

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(self, optimizer, last_epoch)

    def get_lr(self):
        # we will use the first m batches, and set the learning
        # rate to base_lr * m / total_iters

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
"""


def train(resnet, epoch):

    resnet.train()
    for batch_index, (images, labels) in enumerate(train_set):
        if epoch <= args.warm:
            warmup_scheduler.step()
        print('label:', labels.shape)
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_set) + batch_index + 1

        last_layer = list(resnet.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar(
                    'LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar(
                    'LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_set.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in resnet.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch):
    resnet.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_set:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Adduracy: {:.4f}'.format(
        test_loss / len(test_set.dataset),
        correct.float() / len(test_set.dataset)
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss /
                      len(test_set.dataset), epoch)
    writer.add_scalar('Test/Adduracy', correct.float() /
                      len(test_set.dataset), epoch)

    return correct.float() / len(test_set.dataset)


def parser_init():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-net', type=str, default="resnet18", help='net type')
    return parser


if __name__ == '__main__':

    parser = parser_init()
    args = parser.parse_args()

    resnet = get_network()

    # data preprocessing:
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = TrainDataset("train")
    test_set = TrainDataset("test")
    train_set = DataLoader(
        train_set, shuffle=True, num_workers=1, batch_size=8)
    test_set = DataLoader(
        test_set, shuffle=True, num_workers=1, batch_size=8)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=range(10, 100, 10), gamma=0.2)  # learning rate decay
    iter_per_epoch = len(train_set)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    best_add = 0.0
    for epoch in range(1, settings.EPOCH):
        train_scheduler.step(epoch)
        train(epoch)
        add = eval_training(epoch)

        if best_add < add:
            best_add = add
            continue

    # writer.close()
