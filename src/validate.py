# validate
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T


class Validator():

    def __init__(self, resnet,
                 hyper_params,
                 use_cuda,
                 data_loader):
        self.resnet = resnet
        self.hyper_params = hyper_params
        self.use_cuda = use_cuda
        self.data_loader = data_loader
        pass

    def validate(self, SHOW_PIC=False, TTA=False):

        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        accs = []
        for i, data in enumerate(self.data_loader):

            """preprocess"""
            val_x, val_y = data
            val_y = val_y.type(torch.LongTensor)

            """get output"""
            # S x: [batch_size, 3, width, height]
            x = val_x
            if not isinstance(x, torch.Tensor):
                x = T.ToTensor()(x)
            if use_cuda:
                x = x.cuda()
            predict_y = self.resnet(x)
            # S predict_y_cpu: [batch_size, class_num]
            predict_y = predict_y.detach().cpu()

            """calc accuracy"""
            predict_y = np.argmax(predict_y, axis=1)
            acc = torch.sum(predict_y == val_y).type(torch.float)
            acc = acc.numpy() / batch_size
            accs.append(acc)
            pass

        mean_acc = np.mean(np.array(accs))
        return mean_acc

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

        if comment != "":
            plt.text(0, 1, comment, fontsize=14)

        plt.show()
