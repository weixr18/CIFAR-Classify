# train
import os
import gc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


from net import get_network
from data import get_dataset
from validate import Validator

SHOW_NET = False


class Trainer():
    def __init__(self):
        pass

    def setup(self, valid_rate=0.1, use_cuda=True, model_path="", use_exist_dataset=False,
              module_save_dir="", tmp_dir="",
              criterion=None, hyper_params=None,
              FREEZE_PARAM=False, PRETRAINED=False
              ):
        """setup the module"""
        self.train_dataset, self.valid_dataset = get_dataset(
            valid_rate, USE_TRANSFORM=True)

        self.hyper_params = hyper_params
        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=True
        )
        self.valid_data_loader = DataLoader(
            dataset=self.valid_dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

        self.use_cuda = use_cuda
        self.resnet = get_network()
        if PRETRAINED:
            self.resnet.load_state_dict(torch.load(model_path))
        if use_cuda:
            self.resnet = self.resnet.cuda()
        if SHOW_NET:
            from torchsummary import summary
            batch_size = self.hyper_params["batch_size"]
            input_size = self.hyper_params["input_size"][0]
            summary(self.resnet, (3, input_size, input_size), batch_size)

        if(self.hyper_params["optimizer"] == "SGD"):
            self.optimizer = torch.optim.SGD(
                self.resnet.parameters(), lr=self.hyper_params["learning_rate"], momentum=0.99)
        elif (self.hyper_params["optimizer"] == "Adam"):
            self.optimizer = torch.optim.Adam(
                self.resnet.parameters(), lr=self.hyper_params["learning_rate"],
            )

        self.StepLR = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.hyper_params["step_size"], gamma=self.hyper_params["lr_gamma"])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.module_save_dir = module_save_dir
        self.v = Validator(resnet=self.resnet,
                           hyper_params=hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.valid_data_loader)
        pass

    def train(self):
        """train the model"""
        epochs = self.hyper_params["epochs"]
        epoch_lapse = self.hyper_params["epoch_lapse"]
        batch_size = self.hyper_params["batch_size"]
        epoch_save = self.hyper_params["epoch_save"]
        width_out, height_out = self.hyper_params["input_size"]
        prefix = self.hyper_params["name_prefix"]

        for _ in range(epochs):

            total_loss = 0

            for data in tqdm(self.train_data_loader, ascii=True, ncols=120):

                batch_train_x, batch_train_y = data
                batch_train_y = batch_train_y.long()
                if (len(batch_train_x.size()) == 3):
                    batch_train_x = batch_train_x.unsqueeze(1)
                if (len(batch_train_y.size()) == 3):
                    batch_train_y = batch_train_y.unsqueeze(1)

                if self.use_cuda:
                    batch_train_x = batch_train_x.cuda()
                    batch_train_y = batch_train_y.cuda()

                batch_loss = self.train_step(
                    batch_train_x, batch_train_y,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    resnet=self.resnet,
                    width_out=width_out,
                    height_out=height_out,
                    batch_size=batch_size)

                total_loss += batch_loss
                pass

            if (_+1) % epoch_lapse == 0:
                val_acc = self.v.validate()
                print("Total loss in epoch %d : %f , learning rate : %f,  validation accuracy : %f" %
                      (_ + 1, total_loss, self.StepLR.get_lr()[0], val_acc))

            if (_ + 1) % epoch_save == 0:
                name_else = prefix + "-epoch-" + \
                    str(_ + 1) + "-validacc-" + str(val_acc)
                self.save_module(name_else=name_else)
                print("MODULE SAVED.")

            self.StepLR.step()
            pass

        gc.collect()
        pass

    def train_step(self, inputs, labels, optimizer,
                   criterion, resnet, batch_size,
                   width_out, height_out):
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        return loss

    def save_module(self, name_else=""):
        import datetime
        module_save_dir = self.module_save_dir
        filename = 'resnet-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + \
            name_else + '.pth'
        torch.save(self.resnet.state_dict(), module_save_dir + filename)
        pass
