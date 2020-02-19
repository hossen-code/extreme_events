from extreme_events.data_source.rossler import rossler

import numpy as np
from scipy.integrate import odeint

# numerical integration
X_0 = [2, 2, 2]
TIME = np.arange(0, 300, 0.01)


def rossler_dataset(x_0, time_range: np.arange):
    result = odeint(rossler, x_0, time_range)
    x, y, z = result.T

    return x, y, z


if __name__ == "__main__":
    data = rossler_dataset(x_0=X_0, time_range=TIME)