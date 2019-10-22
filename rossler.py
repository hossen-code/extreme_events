"""
Program to plot Rössler attractor
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

#parameters
a = 0.38
b = 0.35
c = 4.5

def rossler(X, t):
    x, y, z = X    
    dx = -y - z
    dy = x + a*y
    dz = b*x - c*z + x*z
    return [dx, dy, dz]

# numerical integration
X0 = [2, 2, 2]
time = np.arange(0, 300, 0.01)
result = odeint(rossler, X0, time)
x, y, z = result.T


# figure
fig = plt.figure(figsize=(13,9))
ax = fig.gca(projection='3d')
ax.set_ylim(-6, 6)
ax.set_xlim(-6, 6)
ax.set_zlim(0, 12)
ax.view_init(20, 160)
ax.set_axis_off()
ax.plot(x,y,z,'magenta')
plt.show()