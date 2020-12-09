# validate
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T


def CUDA(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


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

            """Test time augmentation"""
            # S val_x: [batch_size, 3, width, height]
            if TTA:
                val_x_fh = torch.flip(val_x, dims=[2])
                val_x_fv = torch.flip(val_x, dims=[3])
                val_x_90 = torch.rot90(val_x, 1, dims=(2, 3))
                val_x_180 = torch.rot90(val_x, 2, dims=(2, 3))
                val_x_270 = torch.rot90(val_x, 3, dims=(2, 3))

                val_x_list = [
                    val_x,
                    val_x_fh,
                    val_x_fv,
                    val_x_90,
                    val_x_180,
                    val_x_270,
                ]
            else:
                val_x_list = [val_x]

            """get binary output"""
            # S val_x_list: [6 or 1, batch_size, width, height]

            y_list_cpu = []
            for x in val_x_list:
                # S x: [batch_size, 3, width, height]
                if not isinstance(x, torch.Tensor):
                    x = T.ToTensor()(x)

                # S x: [batch_size, 3, width, height]
                if use_cuda:
                    x = x.cuda()

                """get raw output"""
                predict_y = self.resnet(x)

                """binarization"""
                # S predict_y: [batch_size, class_num]
                predict_y_cpu = predict_y.detach().cpu()

                # S predict_y: [batch_size, class_num]
                y_list_cpu.append(predict_y_cpu)

            """Augmentation vote"""
            # S y_list_cpu: [6 or 1, batch_size, class_num]

            if TTA:
                # S y_list_cpu[n]: [batch_size, class_num]
                predict_y = torch.stack(tuple(y_list_cpu), dim=0)
                predict_y = torch.mean(predict_y, dim=0)

                # S predict_y: [batch_size, class_num]
                predict_y[predict_y > 0.5] = 1
                predict_y[predict_y <= 0.5] = 0

            else:
                predict_y = y_list_cpu[0]

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

        if comment is not "":
            plt.text(0, 1, comment, fontsize=14)

        plt.show()
