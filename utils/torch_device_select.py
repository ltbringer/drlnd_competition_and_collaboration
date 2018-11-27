import torch


def torch_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")