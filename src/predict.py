import imageio
import os
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from net import get_network
from data import PredictSet
from validate import Validator


SHOW_PIC = True


class Predictor():

    def __init__(self, model_path, cell_dir, save_dir,
                 hyper_params, use_cuda):

        self.test_set = PredictSet(cell_dir)
        print("test number:", len(self.test_set))

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

        self.unet = UNet(n_channels=1, n_classes=2,)
        self.unet.load_state_dict(torch.load(model_path))
        if use_cuda:
            self.unet = self.unet.cuda()
        pass

    def predict(self, TTA=True):

        width_out = 628
        height_out = 628

        TTA_KERNEL_SIZE = self.hyper_params["TTA_KERNEL_SIZE"]
        BG_KERNEL_SIZE = self.hyper_params["BG_KERNEL_SIZE"]
        DILATE_ITERATIONS = self.hyper_params["DILATE_ITERATIONS"]
        BIN_THRESHOLD = self.hyper_params["BIN_THRESHOLD"]

        batch_size = self.hyper_params["batch_size"]
        use_cuda = self.use_cuda

        predict_ys = []
        for i, b_val_x in enumerate(self.data_loader):

            """Test time augmentation"""
            # S b_val_x: [batch_size, width, height]
            if TTA:
                b_val_x_fh = torch.flip(b_val_x, dims=[1])
                b_val_x_fv = torch.flip(b_val_x, dims=[2])
                b_val_x_90 = torch.rot90(b_val_x, 1, dims=(1, 2))
                b_val_x_180 = torch.rot90(b_val_x, 2, dims=(1, 2))
                b_val_x_270 = torch.rot90(b_val_x, 3, dims=(1, 2))

                b_val_x_list = [
                    b_val_x,
                    b_val_x_fh,
                    b_val_x_fv,
                    b_val_x_90,
                    b_val_x_180,
                    b_val_x_270,
                ]
            else:
                b_val_x_list = [b_val_x]

            """get binary output"""
            # S b_val_x_list: [6 or 1, batch_size, width, height]

            b_y_list_cpu = []
            for b_x in b_val_x_list:

                # S b_x: [batch_size, width, height]
                if not isinstance(b_x, torch.Tensor):
                    b_x = T.ToTensor()(b_x)
                if (len(b_x.size()) == 3):
                    b_x = b_x.unsqueeze(1)
                elif (len(b_x.size() == 2)):
                    b_x = b_x.unsqueeze(0)
                    b_x = b_x.unsqueeze(1)

                # S b_x: [batch_size, 1, width, height]
                if use_cuda:
                    b_x = b_x.cuda()

                """get raw output"""
                b_predict_y = self.unet(b_x)

                """binarization"""
                # S b_predict_y: [batch_size, 2, width, height]
                b_predict_y_cpu = self.binarization(
                    batch_predict_y=b_predict_y,
                    BIN_THRESHOLD=BIN_THRESHOLD
                ).detach().cpu()

                # S b_predict_y: [batch_size, width, height]
                b_y_list_cpu.append(b_predict_y_cpu)

            """Augmentation vote"""
            # S b_y_list_cpu: [6 or 1, batch_size, width, height]

            if TTA:
                # S b_y_list_cpu[n]: [batch_size, width, height]
                b_y_list_cpu[1] = torch.flip(b_y_list_cpu[1], dims=[1])
                b_y_list_cpu[2] = torch.flip(b_y_list_cpu[2], dims=[2])
                b_y_list_cpu[3] = torch.rot90(
                    b_y_list_cpu[3], 3, dims=(1, 2))
                b_y_list_cpu[4] = torch.rot90(
                    b_y_list_cpu[4], 2, dims=(1, 2))
                b_y_list_cpu[5] = torch.rot90(
                    b_y_list_cpu[5], 1, dims=(1, 2))

                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, TTA_KERNEL_SIZE)

                """Open operation"""
                for n in range(6):
                    b_y_list_cpu[n] = b_y_list_cpu[n].numpy()
                    for j in range(batch_size):
                        b_y_list_cpu[n][j] = cv2.morphologyEx(
                            src=b_y_list_cpu[n][j],
                            op=cv2.MORPH_OPEN,
                            kernel=kernel)
                    b_y_list_cpu[n] = torch.tensor(b_y_list_cpu[n])
                    b_y_list_cpu[n] = b_y_list_cpu[n].unsqueeze(0)

                """Vote"""
                # S b_y_list_cpu[n]: [1, batch_size, width, height]
                b_predict_y = torch.cat(tuple(b_y_list_cpu), dim=0)
                b_predict_y = torch.mean(b_predict_y, dim=0)

                # S b_predict_y: [batch_size, width, height]
                b_predict_y[b_predict_y > 0.5] = 1
                b_predict_y[b_predict_y <= 0.5] = 0

            else:
                b_predict_y = b_y_list_cpu[0]

            """Instance Sparse"""
            # S b_predict_y: [batch_size, width, height]
            b_predict_y = self.instance_sparse(b_predict_y,
                                               BG_KERNEL_SIZE=BG_KERNEL_SIZE,
                                               DILATE_ITERATIONS=DILATE_ITERATIONS)

            for y in b_predict_y:
                y = y.detach().cpu()
                predict_ys.append(y)

            print("Batch %d done." % i)
            pass

        self.save_pics(predict_ys)
        print("Files saved.")
        pass

    def save_pics(self, predict_ys):

        save_dir = self.save_dir
        for i, y in enumerate(predict_ys):
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            full_path = os.path.join(save_dir, 'mask{:0>3d}.tif'.format(i))
            if os.path.exists(full_path):
                os.remove(full_path)
            imageio.imwrite(full_path, y.astype(np.uint16))

        zip_name = "dataset1_" + self.model_path.split("/")[-1].split(".")[-2]
        zip_name = save_dir + "../" + zip_name

        def make_zip(source_dir, output_filename):
            zipf = zipfile.ZipFile(output_filename, 'w')
            pre_len = len(os.path.dirname(source_dir))
            for parent, dirnames, filenames in os.walk(source_dir):
                for filename in filenames:
                    pathfile = os.path.join(parent, filename)
                    arcname = pathfile[pre_len:].strip(
                        os.path.sep)  # 相对路径
                    zipf.write(pathfile, arcname)
            zipf.close()

        # make_zip(save_dir, zip_name)

        pass

    def binarization(self, batch_predict_y, BIN_THRESHOLD=0.6):
        # S b_predict_y: [batch_size, 2, width, height]

        # sqeeze
        batch_predict_y_1 = torch.softmax(batch_predict_y, dim=1)
        batch_predict_y_1 = batch_predict_y_1[:, 1, :, :]
        THRESHOLD_1 = BIN_THRESHOLD
        batch_predict_y_1[batch_predict_y_1 > THRESHOLD_1] = 1
        batch_predict_y_1[batch_predict_y_1 <= THRESHOLD_1] = 0

        batch_predict_y = batch_predict_y_1
        # S b_predict_y: [batch_size, width, height]

        return batch_predict_y

    def instance_sparse(self, batch_predict_y,
                        BG_KERNEL_SIZE=(8, 8),
                        DILATE_ITERATIONS=10):
        """Post process the result."""
        # shape: [batch_size, width, height]

        res = []
        for predict_y in batch_predict_y:

            predict_y = predict_y.numpy().astype(np.uint8) * 255
            predict_y_old = predict_y.copy()

            # sure background area
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, BG_KERNEL_SIZE)
            sure_bg = cv2.dilate(
                predict_y, kernel, iterations=DILATE_ITERATIONS)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(predict_y, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(
                dist_transform, 0.45*dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Change background to 1
            markers = markers + 1
            # Mark the region of unknown with zero
            markers[unknown == 255] = 0

            # watershed
            predict_y_color = cv2.cvtColor(predict_y, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(predict_y_color, markers)
            markers[markers == -1] = 1
            markers = markers - 1

            predict_y = markers.astype(int)
            res.append(predict_y)

        res = torch.Tensor(res)
        return res

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
