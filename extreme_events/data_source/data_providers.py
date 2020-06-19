from collections import namedtuple
from math import floor
from typing import List

import numpy as np
from scipy.integrate import odeint

from extreme_events.data_source.rossler import rossler

data_format = namedtuple("data", ["feature", "target"])


def rossler_dataset_maker(x_0: List[float], time_range: np.arange) -> data_format:
    """
    return the dataset in x, y, z format. z is the target.
    :param x_0: is the initial point [x0, y0, z0]
    :param time_range: is numpy range e.g. np.arange(0, 300, 0.1)
        from 0 to 300, with the step 0.1
    :return: data to train
    """
    result = odeint(rossler, x_0, time_range)
    x, y, z = result.T
    return data_format((x, y), (z,))


def train_test_splitter(data_column: np.ndarray,
                        train_ratio: float):
    """
    for now we're assuming the last member is the target.
    """
    train_size = floor(len(data_column)*train_ratio)
    train_set = data_column[:train_size]
    test_set = data_column[train_size:]

    return train_set, test_set
