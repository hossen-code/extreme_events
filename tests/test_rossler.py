import numpy as np

from extreme_events.data_source.data_providers import rossler_dataset_maker

def test_rossler_dataset_creator():
    x_0 = [2, 2, 2]
    time_range = np.arange(0, 30, 0.1)
    data = rossler_dataset_maker(x_0=x_0, time_range=time_range)
    first_five_x = np.array([2., 1.607831, 1.23093264, 0.8672437,  0.51370104])
    first_five_y = np.array([2., 2.26131837, 2.49356973, 2.69709381, 2.8719731])
    first_five_z = np.array([2., 1.58200549, 1.20471963, 0.88366574, 0.62337719])
    np.testing.assert_allclose(first_five_x, data.feature[0][0:5])
    np.testing.assert_allclose(first_five_y, data.feature[1][0:5])
    np.testing.assert_allclose(first_five_z, data.target[0][0:5])

