import numpy as np
from utils import *

def resample_step(x_bar_t):
    """Ressampling step for the PF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    """

    """STUDENT CODE START"""
    xytheta_indices = np.array([0,1,4])
    weight_index = 5
    xythetas = x_bar_t[xytheta_indices]
    weights = x_bar_t[weight_index]
    x_new, y_new, theta_new = np.random.Generator.choice(
        xythetas,
        p = weights,
        axis = 1) # select by columns
    x_bar_t[xytheta_indices] = np.array([x_new, y_new, theta_new])
    x_est_t = x_bar_t
    """STUDENT CODE END"""

    return x_est_t
