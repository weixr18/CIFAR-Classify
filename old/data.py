import numpy as np
import pandas as pd
import torch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data = np.load("q1_data/train.npy")
        self.data = self.data.reshape([-1, 3, 32, 32])
        label_file = pd.read_csv("q1_data/train1.csv", header=None)
        self.label = label_file.values[1:]
        self.label = self.label.astype(np.uint8)

    def __getitem__(self, index):
        img = self.data[index]
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        label = self.label[index][1]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.data.shape[0]
