
import numpy as np

def rossler(t:float, y:float):
    n = np.round(len(y) / 3) # why len of y
    
    a = 0.38 # 0.2
    b = 0.3 # 0.2
    c = 4.82 # 5.7
    # use numpy array
    x_1 = y[0:n]
    x_2 = y[n:2*n]
    x_3 = y[2*n:3*n]
    
    dy = np.zeros(1, len(y))
    dy[0:n] = - x_1 - x_3
    dy[n:2*n] = x_1 + a * x_2
    dy[2*n:3*n] = b + x_3 * (x_1 - c)