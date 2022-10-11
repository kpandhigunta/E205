import numpy as np
from utils import *
from scipy.stats import multivariate_normal

STD = 1
X_VAR = 0.5
Y_VAR = 0.5
THETA_VAR = 0.2
NUM_PARTICLES = 100
MU = np.zeros(NUM_PARTICLES)

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
    for i in range(theta_n.shape[0]):
        theta_n[i] = wrap_to_pi(theta_n[i])
    x_bar_t = np.array([x_n, y_n, vx_n, vy_n, theta_n, weight_prev])
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
    x_n, y_n, vx_n, vy_n, theta_n, weight_prev = propogate_state(x_t_prev, u_t)
    x_noisy = x_n + np.random.multivariate_normal(MU, X_VAR * np.identity(NUM_PARTICLES))
    y_noisy = y_n + np.random.multivariate_normal(MU, Y_VAR * np.identity(NUM_PARTICLES))
    theta_n = theta_n + np.random.multivariate_normal(MU, THETA_VAR * np.identity(NUM_PARTICLES))
    x_bar_t = np.array([x_noisy, y_noisy, vx_n, vy_n, theta_n, weight_prev])

    # Correction Step
    z_x, z_y, z_theta = z_t
    prob_z_given_x_t = multivariate_normal(
        mean=np.array([z_x, z_y, z_theta]),
        cov=np.array([[X_VAR, 0, 0],
                      [0, Y_VAR, 0],
                      [0, 0, THETA_VAR]]))
    xytheta_indices = np.array([0,1,4])
    weight_n = prob_z_given_x_t.pdf(x_bar_t[xytheta_indices].T)
    x_bar_t = np.array([x_noisy, y_noisy, vx_n, vy_n, theta_n, weight_n])
    """STUDENT CODE END"""

    return x_bar_t
