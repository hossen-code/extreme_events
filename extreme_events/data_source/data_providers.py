from collections import namedtuple
from math import floor
from typing import List

import numpy as np
from scipy.integrate import odeint

from extreme_events.data_source.rossler import rossler

data_format = namedtuple("data", ["feature", "target"])


def rossler_dataset_maker(x_0: List[float], time_range: np.arange) -> np.ndarray:
    """
    return the dataset in x, y, z format of float32 numbers. z is the target.
    :param x_0: is the initial point [x0, y0, z0]
    :param time_range: is numpy range e.g. np.arange(0, 300, 0.1)
        from 0 to 300, with the step 0.1
    :return: data to train, data.shape is (a, b, c) which a is the number of batches
    b is the time series (values through time), c is the number of variates + target
    (for now we assume, the last one is the target as our problem is single-target).
    the array shape, is to comply with standard deep learning models.
    """
    result = odeint(rossler, x_0, time_range)
    x, y, z = result.T
    stacked_and_converted = np.float32(np.stack([x, y, z], axis=1))
    final_shaped_array = np.expand_dims(stacked_and_converted, axis=0)
    return final_shaped_array


def train_test_splitter(data_column: np.ndarray,
                        train_ratio: float):
    """
    TODO: this needs to change, to be vertical split (or have the option)
    for now we're assuming the last member is the target.
    """
    train_size = floor(len(data_column)*train_ratio)
    train_set = data_column[:train_size]
    test_set = data_column[train_size:]

    return train_set, test_set


def threshold_binarizer(array: np.ndarray,
                        threshold: float):
    """
    Turns an array of floats into a binary array (0. and 1.) where
    the value smaller than threshold returns 0, otherwise 1.
    """
    return (array > threshold) * np.ones(np.shape(array))