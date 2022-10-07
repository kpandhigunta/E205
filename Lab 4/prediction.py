import numpy as np
from utils import *
from scipy.stats import norm

MU = np.zeros(1000)
STD = 1
X_STD = 0.5
Y_STD = 0.5
THETA_STD = 0.2

def propogate_state(x_t_prev, u_t_noiseless):
    """Propagate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    """STUDENT CODE START"""
    x_n_prev, y_n_prev, vx_n_prev, vy_n_prev, theta_n_prev, weight_prev = x_t_prev
    a_x_prime, omega = u_t_noiseless
    x_n = x_n_prev + vx_n_prev * DT
    y_n = y_n_prev + vy_n_prev * DT
    vx_n = vx_n_prev + a_x_prime * np.cos(theta_n_prev) * DT
    vy_n = vy_n_prev + a_x_prime * np.sin(theta_n_prev) * DT
    theta_n = theta_n_prev + omega * DT
    x_bar_t = np.array([x_n, y_n, vx_n, vy_n, weight_prev])
    """STUDENT CODE END"""
    return x_bar_t


def prediction_step(x_t_prev, u_t, z_t):
    """Compute the prediction and correction (re-weighting) for the PF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    z_t (np.array)              -- the current measurement

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    """

    """STUDENT CODE START"""
    # Prediction step
    x_n, y_n, vx_n, vy_n, weight_prev = propogate_state(x_t_prev, u_t)
    x_noisy = x_n + np.random.multivariate_normal(MU, STD * np.identity(1000))
    y_noisy = y_n + np.random.multivariate_normal(MU, STD * np.identity(1000))
    x_bar_t = np.array([x_noisy, y_noisy, vx_n, vy_n, weight_prev])

    # find z_t
    theta_t = wrap_to_pi(yaw_lidar[t])
    z_t = np.array([ 5 - (y_lidar[t] * np.cos(theta_t) + x_lidar[t] * np.sin(theta_t)),
                    -5 - (y_lidar[t] * np.sin(theta_t) - x_lidar[t] * np.cos(theta_t)),
                    theta_t])
    z_t_log[:,t] = z_t

    # correction_step

    
    """STUDENT CODE END"""

    return x_bar_t
