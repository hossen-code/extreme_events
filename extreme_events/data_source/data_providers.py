from rossler import rossler

import numpy as np
from scipy.integrate import odeint

# numerical integration
x_0 = [2, 2, 2]
time = np.arange(0, 300, 0.01)
result = odeint(rossler, x_0, time)
x, y, z = result.T

