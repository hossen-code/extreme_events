from collections import namedtuple
from math import floor
from typing import List

import numpy as np
from scipy.integrate import odeint

from extreme_events.data_source.rossler import rossler

data_format = namedtuple("data", ["feature", "target"])


def rossler_dataset_maker(x_0: List[float],
                          start_time: float,
                          end_time: float,
                          time_step: float,
                          target_advance_time: float = 0.0,
                          target_threshold_val=None) -> np.ndarray:
    """
    return the dataset in x, y, z format of float32 numbers. z is the target.
    :param x_0: is the initial point [x0, y0, z0]
    :param start_time:
    :param end_time:
    :param time_step:
    :param target_advance_time:
    :param target_threshold_val:
    :return: data to train, data.shape is (a, b, c) which a is the number of batches
    b is the time series (values through time), c is the number of variates + target
    (for now we assume, the last one is the target as our problem is single-target).
    the array shape, is to comply with standard deep learning models.
    """
    time_range = np.arange(start_time, end_time+target_advance_time, time_step)
    # calculate for the whole time (real time + advance time)
    result = odeint(rossler, x_0, time_range)
    x, y, z = result.T
    real_length = len(np.arange(start_time, end_time, time_step))
    shift_length = len(time_range) - real_length
    x = x[:real_length]
    y = y[:real_length]
    z = z[shift_length:]
    if target_threshold_val is not None:
        if not isinstance(target_threshold_val, float):
            raise TypeError(f"target_threshold_binarizer must be a float, given {type(target_threshold_val)}")
        z = threshold_binarizer(z, target_threshold_val)
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


def threshold_binarizer(array: np.ndarray, threshold: float):
    """
    Turns an array of floats into a binary array (0. and 1.) where
    the value smaller than threshold returns 0, otherwise 1.
    """
    return (array > threshold) * np.ones(np.shape(array), dtype=int)


def if_flips_in_next_n_steps(array: np.array, threshold: float, n_time_steps):
    """
    Returns an encoding that shows if the value of next `n_time_steps` has flipped
    compared to value at given time. Flipping means if went above threshold or not.
    """
    res = []
    binary_array = threshold_binarizer(array, threshold)
    for i in range(len(array) - n_time_steps):
        if binary_array[i]:
            if np.all(binary_array[i+1:i+n_time_steps+1]):
                res.append(0)
            else:
                res.append(1)
        if not binary_array[i]:
            if np.any(binary_array[i+1:i+n_time_steps+1]):
                res.append(1)
            else:
                res.append(0)

    return np.array(res)
