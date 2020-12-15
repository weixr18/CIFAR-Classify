# net
import torch
from torchvision import models


def get_network(mission=1):
    if(mission == 1):
        resnet = models.resnet18(pretrained=False, num_classes=20)
    else:
        resnet = models.resnet18(pretrained=False, num_classes=100)
    return resnet
