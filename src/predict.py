import imageio
import os
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from net import get_network
from data import get_dataset
from validate import Validator


SHOW_PIC = True


class Predictor():

    def __init__(self, model_path, save_dir, hyper_params, use_cuda, mission=1):

        _, self.dataset = get_dataset(
            valid_rate=1, USE_TRANSFORM=False, mission=mission)

        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.test_set,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )
        self.save_dir = save_dir
        self.model_path = model_path
        self.use_cuda = use_cuda

        self.resnet = get_network(mission=mission)
        self.resnet.load_state_dict(torch.load(module_path))
        if use_cuda:
            self.unet = self.unet.cuda()
        pass

    def predict(self, TTA=True):

        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        predicts = []
        for i, data in enumerate(self.data_loader):

            """preprocess"""
            # S val_x: [batch_size, 3, width, height]
            val_x, val_y = data
            val_y = val_y.type(torch.LongTensor)

            """get output"""
            x = val_x
            # S x: [batch_size, 3, width, height]
            if not isinstance(x, torch.Tensor):
                x = T.ToTensor()(x)
            if use_cuda:
                x = x.cuda()

            """get raw output"""
            predict_y = self.resnet(x)

            # S predict_y: [batch_size, class_num]
            predict_y = predict_y.detach().cpu()
            # S predict_y: [batch_size]
            predict_y = np.argmax(predict_y, axis=1)
            predicts.append(predict_y)
            pass

        """write into csv"""

        self.write_csv(predicts)
        print("Files saved.")
        pass

    def write_csv(self, predicts):
        pass

    def show_pic(self, picA, picB, picC=None,
                 is_gray=(True, False, False), comment=""):
        plt.subplot(1, 3, 1)
        plt.title("x")
        if is_gray[0]:
            plt.imshow(picA, cmap='gray')
        else:
            plt.imshow(picA)

        plt.subplot(1, 3, 2)
        plt.title("GT")
        if is_gray[1]:
            plt.imshow(picB, cmap='gray')
        else:
            plt.imshow(picB)

        if picC is not None:
            plt.subplot(1, 3, 3)
            plt.title("Predict")
            if is_gray[2]:
                plt.imshow(picC, cmap='gray')
            else:
                plt.imshow(picC)

        if comment is not "":
            plt.text(0, 1, comment, fontsize=14)

        plt.show()
