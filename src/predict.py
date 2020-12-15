import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from net import get_network
from data import get_test_set
from validate import Validator


SHOW_PIC = True


class Predictor():

    def __init__(self, model_path, save_dir, hyper_params, use_cuda, mission=1):

        self.dataset = get_test_set(mission=mission)
        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )
        self.save_dir = save_dir
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.mission = mission

        self.resnet = get_network(mission=mission)
        self.resnet.load_state_dict(torch.load(model_path))
        if use_cuda:
            self.resnet = self.resnet.cuda()
        pass

    def predict(self, TTA=True):

        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        i = 0
        predicts = np.ndarray([len(self.dataset)], dtype=np.int32)
        for data in tqdm(self.data_loader, ascii=True, ncols=120):

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
            predicts[i * batch_size:(i + 1) * batch_size] = predict_y.numpy()
            i += 1
            pass

        """write into csv"""

        self.write_csv(predicts, mission=self.mission)
        print("Files saved.")
        pass

    def write_csv(self, predicts, mission=1):
        index = np.arange(0, predicts.shape[0], 1, dtype=np.uint8)
        predicts = np.vstack([index, predicts]).T
        np.savetxt('res/'+str(mission)+'.csv', predicts, delimiter=',')
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
