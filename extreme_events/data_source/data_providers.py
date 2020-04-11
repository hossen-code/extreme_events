from collections import namedtuple
from math import floor

import numpy as np
from scipy.integrate import odeint

from extreme_events.data_source.rossler import rossler

X_0 = [2, 2, 2]
TIME = np.arange(0, 300, 0.01)

data = namedtuple("data", ["feature", "target"])


def rossler_dataset(x_0, time_range: np.arange):
    result = odeint(rossler, x_0, time_range)
    x, y, z = result.T
    return data((x, y), (z,))


def train_test_splitter(data_column: np.ndarray,
                        train_ratio: float):
    """
    for now we're assuming the last member is the target.
    """
    train_size = floor(len(data_column)*train_ratio)
    train_set = data_column[:train_size]
    test_set = data_column[train_size:]

    return train_set, test_set


if __name__ == "__main__":
    data = rossler_dataset(x_0=X_0, time_range=TIME)
    pass