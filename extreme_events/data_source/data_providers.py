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
    :param start_time: starting time fo the rossler data
    :param end_time: ending time, duh!
    :param time_step: the time resoloution to generate dataset
    :param target_advance_time: how long in advance you want to consider the prediction
    :param target_threshold_val: the threshold that above that the value is considered extreme
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
            raise TypeError(f"target_threshold_binarizer must be a float, given "
                            f"{type(target_threshold_val)}")
        z = threshold_binarizer(z, target_threshold_val)
    stacked_and_converted = np.float32(np.stack([x, y, z], axis=1))
    final_shaped_array = np.expand_dims(stacked_and_converted, axis=0)
    return final_shaped_array


def train_test_splitter(data_column: np.ndarray,
                        train_ratio: float):
    """
    TODO: test for n-dimensional, should be good for 1d
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


def if_flips_in_next_n_steps(inp_array: np.array, *, threshold: float, n_time_steps: int,
                             padding=True):
    """
    Returns an encoding that shows if the value of next `n_time_steps` has flipped
    compared to value at given time. Flipping means if went above threshold or went below threshold.
    If the state has changed.

    If the value of array[i] is already above threshold and all next n_time_steps also above
    threshold, the encoded value of output[i] is 0, otherwise, if it goes below threshold in the
    next n_time_steps it output[i] is 1.
    Similarly, if the value of array[i] is below threshold, and it goes above within the next
    n_time_steps, the value of output[i] is 1.
    """
    res = []
    inp_arr_shape = inp_array.shape
    if len(inp_arr_shape) > 2 or inp_arr_shape[0] != 1:
        raise TypeError(f"the shape of input array must be (1, n), received {inp_arr_shape}")
    flat_inp_array = inp_array.flatten()
    binary_array = threshold_binarizer(flat_inp_array, threshold)
    for i in range(inp_arr_shape[1] - n_time_steps):
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
    if padding:
        res += [res[-1]] * n_time_steps
        res = np.array(res).reshape(inp_arr_shape)
    else:
        # note that the output shape will not be the same as input
        res = np.array(res).reshape(1, len(res))

    return res
