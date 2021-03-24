from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
