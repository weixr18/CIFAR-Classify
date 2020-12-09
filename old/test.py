import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import csv
import pandas as pd


if __name__ == "__main__":
    """
    print(torch.cuda.is_available())
    resnet = models.resnet18(pretrained=True)
    device = torch.device("cuda")
    resnet = resnet.to(device)
    """
    """
    train = np.load("q1_data/train.npy")
    print(train.shape)
    train = train.reshape([50000, 3, 32, 32])
    sample = train[0]
    sample = np.swapaxes(sample, 0, 1)
    sample = np.swapaxes(sample, 1, 2)
    plt.imshow(sample)
    plt.show()
    """

    pass
