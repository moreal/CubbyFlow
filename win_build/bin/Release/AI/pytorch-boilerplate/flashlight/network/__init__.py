import inspect

from ..utils import func

import torch
import torchvision

from . import pointnet
class FCNet(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(FCNet, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_size, 256),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, out_size)
        )

        self.out_size= out_size
    
    def forward(self, x):
        return self.net(x.float()).view(-1, int(self.out_size/3), 3)

def check_network_option(network, network_option_dict):
    if len(network_option_dict.keys()) > 0:
        func.function_arg_checker(NETWORK_DICT[network], network_option_dict)
    return network_option_dict


"""Network Define"""
# Add {"Network Name" : and nn.Module without initalize}
def _get_squeezenet(num_classes, version: str = "1_0", pretrained=False, progress=True):
    VERSION = {"1_0": torchvision.models.squeezenet1_0, "1_1": torchvision.models.squeezenet1_1}

    return VERSION[version](pretrained=pretrained, progress=progress, num_classes=num_classes)

def get_pt3d(target_output):
    return pointnet.PointNetDenseCls(target_output=target_output)

def get_fc3d(in_size, out_size):
    return FCNet(in_size, out_size)

NETWORK_DICT = {
    "squeezenet": _get_squeezenet,
    "fc_pt": get_fc3d,
    "pc_pt": get_pt3d
}

def get_network(network_name, network_opt):
    return NETWORK_DICT[network_name](**network_opt)
