# net
import torch
from torchvision import models


def get_network():
    resnet = models.resnet18(pretrained=False, num_classes=20)
    return resnet
