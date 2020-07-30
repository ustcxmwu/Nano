# -*- encoding: utf-8 -*-

import time
from collections import namedtuple

import torch.nn as nn

func_call_dict = dict()
FuncCallInfo = namedtuple('FuncCallInfo', ['call_cnt', 'cur_call_time', 'avg_call_time'])


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


class ProfileUtils(object):
    """
    Utils for performance profile.
    """

    @staticmethod
    def update_call_info(func_info, call_time):
        func_info = func_info._replace(
            cur_call_time=call_time,
            avg_call_time=(func_info.avg_call_time*func_info.call_cnt+call_time) / (func_info.call_cnt + 1),
            call_cnt=func_info.call_cnt + 1
        )
        return func_info

    @staticmethod
    def timeit(**info):
        """
        A wrapper for profiling the func, time unit is ms.
        Args:
            **info:  dict with key 'class_name' and 'interval', class_name used to define the name of the class of
            the called function, interval used to print information every interval calls.

        Returns: None

        """
        def func_wrapper(method):
            def time_func(*args, **kwargs):
                ts = time.time()
                result = method(*args, **kwargs)
                te = time.time()
                global func_call_dict
                func_key = (info["class_name"] + '.' if "class_name" in info else "") + method.__name__
                func_info = func_call_dict.setdefault(func_key, FuncCallInfo(0, 0.0, 0.0))
                func_info = ProfileUtils.update_call_info(func_info, (te - ts) * 1000)
                func_call_dict[func_key] = func_info

                interval = info["interval"] if ("interval" in info and info["interval"] > 0) else 1
                if func_info.call_cnt % interval == 0:
                    print("{}:{}".format(func_key, func_info))
                return result

            return time_func
        return func_wrapper



