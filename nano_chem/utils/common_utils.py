# -*- encoding: utf-8 -*-

import time
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.platform import gfile
import torch.nn as nn
import os
from typing import List

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
            avg_call_time=(func_info.avg_call_time * func_info.call_cnt + call_time) / (func_info.call_cnt + 1),
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


class ModelUtils(object):
    """
    Utils for model convert and validation
    """

    @staticmethod
    def freeze_tb_model_to_pb(input_checkpoint, output_pb_file, output_none_names):
        if output_none_names is None:
            output_none_names = ["ArgMax"]
        saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=True)
        graph = tf.compat.v1.get_default_graph()
        input_graph_def = graph.as_graph_def()

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, input_checkpoint)
            for tensor in sess.graph.get_operations():
                print(tensor.name, tensor.values())
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_none_names=output_none_names
            )

            with tf.qfile.GFile(output_pb_file, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("{} ops in the final graph.".format(len(output_graph_def.node)))
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def convert_pb_to_tflite(pb_file, lite_file, input_node_names: List[str] = None,
                             output_node_names: List[str] = None):
        if input_node_names is None:
            input_node_names = ["X0", "legal_actions"]
        if output_node_names is None:
            output_node_names = ["inference"]
        converter = tf.contrib.lite.TocoConverter.from_frozen_file(pb_file, input_node_names, output_node_names)
        tflite_model = converter.convert()
        with open(lite_file, "wb") as f:
            f.write(tflite_model)

    @staticmethod
    def get_tf_sess_from_pb(pb_file, write_tensor: bool = False):
        sess = tf.compat.v1.Session()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        sess.run(tf.compat.v1.global_variables_initializer())

        tensors = []
        for tensor in tf.contrib.graph_editor.get_tensors(tf.compat.v1.get_default_graph()):
            print(tensor)
            if "read" in tensor.name or "reduction_indices" in tensor.name or "StopGradient" in tensor.name or \
                    "output" in tensor.name:
                tensors.append(str(tensor) + "\n")

        if write_tensor:
            file = os.path.splitext(pb_file)[0] + ".tensor"
            with open(file, mode='w') as f:
                f.writelines(tensors)

        return sess

    @staticmethod
    def inference_validate(input_name, legal_name, feed_dict, inference_name, expected_action, pb_file):
        sess = ModelUtils.get_tf_sess_from_pb(pb_file)
        input = sess.graph.get_tensor_by_name(input_name + ":0")
        legal = sess.graph.get_tensor_by_name(legal_name + ":0")
        inference = sess.graph.get_tensor_by_name(inference_name + ":0")
        action = sess.run(inference, feed_dict={input: feed_dict[input_name], legal: feed_dict[legal_name]})
        if action == expected_action:
            print("test pass")

        return action
