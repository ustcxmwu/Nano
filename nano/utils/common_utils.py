# -*- encoding: utf-8 -*-

import torch.nn as nn


class TorchUtils(object):
    """
    Utils for pytorch.
    """

    def __init__(self):
        pass

    @staticmethod
    def count_params(model: nn.Module):
        """
        count the number of parameters of pytorch model.

        Args:
            model: python model

        Returns: the number of parameters

        """
        params_count = 0
        for param in model.parameters():
            params_count += param.view(-1).size()[0]
        return params_count

