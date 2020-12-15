import torch
from torch.utils.data import Dataset, DataLoader

from net import get_network
from data import get_dataset, get_test_set
from validate import Validator


class Tester():

    def __init__(self, module_path, hyper_params,
                 use_cuda, test_rate=1.0,
                 USE_EXIST_RES=False, mission=1):

        print("Test rate:", test_rate)
        _, self.dataset = get_dataset(
            valid_rate=test_rate, USE_TRANSFORM=False, mission=mission)
        print("test number:", len(self.dataset))

        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

        self.resnet = get_network(mission=mission)
        self.resnet.load_state_dict(torch.load(module_path))
        if use_cuda:
            self.resnet = self.resnet.cuda()

        self.v = Validator(resnet=self.resnet,
                           hyper_params=hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.data_loader)

    def test(self, SHOW_PIC=False, TTA=False):
        return self.v.validate(SHOW_PIC=SHOW_PIC, TTA=TTA)
    pass


class SetTester():

    def __init__(self, module_path, hyper_params, use_cuda, mission=1):

        self.dataset = get_test_set(mission=mission)
        print("test number:", len(self.dataset))

        self.hyper_params = hyper_params
        self.data_loader = DataLoader(
            dataset=self.dataset,
            num_workers=self.hyper_params["threads"],
            batch_size=self.hyper_params["batch_size"],
            shuffle=False
        )

        self.resnet = get_network()
        self.resnet.load_state_dict(torch.load(module_path))
        if use_cuda:
            self.resnet = self.resnet.cuda()

        self.v = Validator(resnet=self.resnet,
                           hyper_params=hyper_params,
                           use_cuda=use_cuda,
                           data_loader=self.data_loader)

    def test(self, SHOW_PIC=False, TTA=False):
        return self.v.validate(SHOW_PIC=SHOW_PIC, TTA=TTA)
    pass
