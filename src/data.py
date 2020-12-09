# data
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


data_transforms = transforms.Compose([
    # transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        super(Dataset, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]

        if self.transform is not None:
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2).astype('uint8')
            img = Image.fromarray(img).convert('RGB')
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.data.shape[0]


def get_data_array():

    data = np.load("q1_data/train.npy")
    data = data.reshape([-1, 3, 32, 32])
    data = data.astype(np.float32)

    label_file = pd.read_csv("q1_data/train1.csv", header=None)
    label = label_file.values[1:, 1]
    label = label.astype(np.float32)

    return data, label


def get_dataset(valid_rate, USE_TRANSFORM=True):
    """get split train & validation datasets"""
    data, label = get_data_array()

    # separate the lists according to valid_rate
    sample_size = data.shape[0]
    valid_size = int(sample_size * valid_rate)
    valid_index = np.random.choice(
        a=sample_size, size=valid_size, replace=False, p=None)
    train_index = set(range(sample_size)) - set(valid_index)
    train_index = np.array(list(train_index))

    valid_data = data[valid_index, :]
    valid_label = label[valid_index]
    train_data = data[train_index, :]
    train_label = label[train_index]

    if USE_TRANSFORM:
        trans = data_transforms
    else:
        trans = None

    # get the Dataset objects
    train_dataset = Dataset(train_data, train_label, trans)
    valid_dataset = Dataset(valid_data, valid_label, trans)

    return train_dataset, valid_dataset


def get_test_set(mission=1):
    data = np.load("q1_data/test.npy")
    data = data.reshape([-1, 3, 32, 32])
    data = data.astype(np.float32)
    label = np.load("q1_data/test_res/test_label" + str(mission) + ".npy")

    test_dataset = Dataset(data, label)
    return test_dataset
