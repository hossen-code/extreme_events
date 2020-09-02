import numpy as np

from extreme_events.data_source.data_providers import rossler_dataset_maker, threshold_binarizer


def test_rossler_dataset_creator():
    x_0 = [2, 2, 2]
    time_range = np.arange(0, 30, 0.1)
    data = rossler_dataset_maker(x_0=x_0, start_time=0.0, end_time=30.0, time_step=0.1)
    first_five_x = np.array([2., 1.607831, 1.23093264, 0.8672437,  0.51370104])
    first_five_y = np.array([2., 2.26131837, 2.49356973, 2.69709381, 2.8719731])
    first_five_z = np.array([2., 1.58200549, 1.20471963, 0.88366574, 0.62337719])
    np.testing.assert_allclose(first_five_x, data[0][0:5, [0]].reshape(5))
    np.testing.assert_allclose(first_five_y, data[0][0:5, [1]].reshape(5))
    np.testing.assert_allclose(first_five_z, data[0][0:5, [2]].reshape(5))


def test_threshold_binarizier():
    x_0 = np.array([1.1, 2.3, 34.5, -2.0, 4.5])
    res = np.array([0, 1, 1, 0, 1])
    threshold = 1.8
    np.testing.assert_allclose(threshold_binarizer(x_0, threshold), res)