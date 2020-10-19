"""
Program to plot Rössler attractor
"""
from matplotlib import pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D # this is for necessary for plot


def rossler(x_0, t):
    """
    Parameters
    __________
     x_0: initial x (not sure actually)
     t: is needed for algebraic operations.
    """
    a = 0.38
    b = 0.35
    c = 4.5
    x, y, z = x_0
    dx = -y - z
    dy = x + a * y
    dz = b * x - c * z + x * z
    return [dx, dy, dz]


# figure
# fig = plt.figure(figsize=(13, 9))
# ax = fig.gca(projection='3d')
# ax.set_ylim(-6, 6)
# ax.set_xlim(-6, 6)
# ax.set_zlim(0, 12)
# ax.view_init(20, 160)
# ax.set_axis_off()
# ax.plot(x, y, z, 'magenta')
# plt.show()
