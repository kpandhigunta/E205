"""
Authors: Sean Wu, Kaanthi Pandhigunta, Andrew Q. Pham (template)
Email: sywu@g.hmc.edu, kpandhigunta@g.hmc.edu, apham@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Extended Kalman Filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 3
    Student code version with parts omitted.
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path

HEIGHT_THRESHOLD = 0.0  # meters
GROUND_HEIGHT_THRESHOLD = -.4  # meters
DT = 0.1
X_LANDMARK = 5.  # meters
Y_LANDMARK = -5.  # meters
EARTH_RADIUS = 6.3781E6  # meters
CCW = -1

def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of floats
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(float(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """

    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (float)    -- latitude coordinate
    lon_gps     (float)    -- longitude coordinate
    lat_origin  (float)    -- latitude coordinate of your chosen origin
    lon_origin  (float)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (float)          -- the converted x coordinate
    y_gps (float)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propogate_state(x_t_prev, u_t):
    """Propogate/predict the state based on chosen motion model

    Parameters:
    x_t_prev (np.array)  -- the previous state estimate
    u_t (np.array)       -- the current control input

    Returns:
    x_bar_t (np.array)   -- the predicted state
    """
    """STUDENT CODE START"""
    G_x_t = calc_prop_jacobian_x(x_t_prev, u_t)
    G_u_t = calc_prop_jacobian_u(x_t_prev, u_t)

    x_bar_t = G_x_t * x_t_prev + G_u_t * u_t
    """STUDENT CODE END"""

    return x_bar_t

def getX(x_t): return x_t[0]
def getY(x_t): return x_t[1]
def getXdot(x_t): return x_t[2]
def getYdot(x_t): return x_t[3]
def getYaw(x_t): return x_t[4]

def calc_prop_jacobian_x(x_t_prev, u_t):
    """Calculate the Jacobian of your motion model with respect to state

    Parameters:
    x_t_prev (np.array) -- the previous state estimate (x, xdot, y, ydot, yaw)
    u_t (np.array)      -- the current control input

    Returns:
    G_x_t (np.array)    -- Jacobian of motion model wrt to x
    """
    """STUDENT CODE START"""
    a_x_prime = u_t[0]
    a_y_prime = u_t[1]
    theta_t_prev = wrap_to_pi(x_t_prev[4])
    G_x_t = np.empty((5, 5))  # add shape of matrix
    G_x_t[0] = np.array([1, 0, DT, 0, -1 * a_x_prime * np.sin(theta_t_prev) * DT * DT])
    G_x_t[1] = np.array([0, 1, 0, DT, a_x_prime * np.cos(theta_t_prev) * DT * DT])
    G_x_t[2] = np.array([1, 0, 0, 0, -1 * a_x_prime * np.sin(theta_t_prev) * DT])
    G_x_t[3] = np.array([0, 1, 0, 0, a_x_prime * np.cos(theta_t_prev) * DT])
    G_x_t[4] = np.array([0, 0, 0, 0, 1])
    # G_x_t[0] = np.array([1, 0, DT, 0, 1/2 * DT**2 * ((a_x_prime * np.cos(theta_t)) - (a_y_prime * np.sin(theta_t)))])
    # G_x_t[1] = np.array([0, 1, 0, DT, 1/2 * DT**2 * (-1*(a_x_prime * np.sin(theta_t)) + (a_y_prime * np.cos(theta_t)))])
    # G_x_t[2] = np.array([1, 0, 0, 0, 0])
    # G_x_t[3] = np.array([0, 1, 0, 0, 0])
    # G_x_t[4] = np.array([0, 0, 0, 0, 1])
    """STUDENT CODE END"""

    return G_x_t


def calc_prop_jacobian_u(x_t_prev, u_t):
    """Calculate the Jacobian of motion model with respect to control input

    Parameters:
    x_t_prev (np.array)     -- the previous state estimate
    u_t (np.array)          -- the current control input

    Returns:
    G_u_t (np.array)        -- Jacobian of motion model wrt to u
    """

    """STUDENT CODE START"""
    G_u_t = np.zeros((5, 3))  # add shape of matrix
    theta_t_prev = wrap_to_pi(x_t_prev[4])
    G_u_t[0] = np.array([DT * DT * np.cos(theta_t_prev), 0, 0])
    G_u_t[1] = np.array([0, DT * DT * np.sin(theta_t_prev), 0])
    G_u_t[2] = np.array([DT * np.cos(theta_t_prev), 0, 0])
    G_u_t[3] = np.array([0, DT * np.sin(theta_t_prev), 0])
    G_u_t[3] = np.array([0, 0, CCW * DT])
    # G_u_t[0] = np.array([1/2 * DT**2 * np.sin(theta_t), 1/2 * DT**2 * np.cos(theta_t), 0])
    # G_u_t[1] = np.array([1/2 * DT**2 * np.cos(theta_t), 1/2 * DT**2 * np.sin(theta_t), 0])
    # G_u_t[2] = np.array([DT * np.sin(theta_t), DT * np.cos(theta_t), 0])
    # G_u_t[3] = np.array([DT * np.cos(theta_t), DT * np.sin(theta_t), 0])
    # G_u_t[3] = np.array([0, 0, DT])
    """STUDENT CODE END"""

    return G_u_t


def prediction_step(x_t_prev, u_t, sigma_x_t_prev):
    """Compute the prediction of EKF

    Parameters:
    x_t_prev (np.array)         -- the previous state estimate
    u_t (np.array)              -- the control input
    sigma_x_t_prev (np.array)   -- the previous variance estimate

    Returns:
    x_bar_t (np.array)          -- the predicted state estimate of time t
    sigma_x_bar_t (np.array)    -- the predicted variance estimate of time t
    """

    """STUDENT CODE START"""
    # Covariance matrix of control input
    sigma_u_t = np.identity(3)  # add shape of matrix
    G_x_t = calc_prop_jacobian_x(x_t_prev, u_t)
    G_u_t = calc_prop_jacobian_u(x_t_prev, u_t)
    x_bar_t = G_x_t @ x_t_prev + G_u_t @ u_t
    sigma_x_bar_t = G_x_t @ sigma_x_t_prev @ G_x_t.T + G_u_t @ sigma_u_t @ G_u_t.T
    """STUDENT CODE END"""

    return [x_bar_t, sigma_x_bar_t]


def calc_meas_jacobian(x_bar_t):
    """Calculate the Jacobian of your measurment model with respect to state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    H_t (np.array)      -- Jacobian of measurement model
    """
    """STUDENT CODE START"""
    H_t = np.zeros((3, 5))
    for i,j in [(0,0), (1,1), (2,4)]:
        H_t[i,j] = 1

    # xlidar = x_bar_t[0]
    # ylidar = x_bar_t[1]
    # theta = x_bar_t[2]
    # H_t[:,4] = np.array([
    #     ylidar * np.sin(theta) - xlidar * np.cos(theta),
    #     -1 * ylidar * np.sin(theta) - ylidar * np.cos(theta),
    #     1
    # ]).T
    """STUDENT CODE END"""

    return H_t


def calc_kalman_gain(sigma_x_bar_t, H_t):
    """Calculate the Kalman Gain

    Parameters:
    sigma_x_bar_t (np.array)  -- the predicted state covariance matrix
    H_t (np.array)            -- the measurement Jacobian

    Returns:
    K_t (np.array)            -- Kalman Gain
    """
    """STUDENT CODE START"""
    # Covariance matrix of measurments
    sigma_z_t = np.identity(3)
    inverse = np.linalg.inv(H_t @ sigma_x_bar_t @ H_t.T + sigma_z_t)
    K_t = sigma_x_bar_t @ H_t.T @ inverse
    """STUDENT CODE END"""

    return K_t


def calc_meas_prediction(x_bar_t):
    """Calculate predicted measurement based on the predicted state

    Parameters:
    x_bar_t (np.array)  -- the predicted state

    Returns:
    z_bar_t (np.array)  -- the predicted measurement
    """

    """STUDENT CODE START"""
    H_t = calc_meas_jacobian(x_bar_t)
    xytheta_states = H_t @ x_bar_t
    z_bar_t = xytheta_states

    # theta_t = getYaw(x_bar_t)
    # lidarTransform = np.array([
    #     -1*np.sin(theta_t), -1*np.cos(theta_t), 0,
    #     np.cos(theta_t), -1*np.sin(theta_t), 0,
    #     0, 0, 1]).reshape((3,3))
    # inverse = np.linalg.inv(lidarTransform)
    # constants = np.array([5, -5, 0])
    # z_bar_t = inverse @ (xytheta_states.T - constants.T)
    """STUDENT CODE END"""

    return z_bar_t


def correction_step(x_bar_t, z_t, sigma_x_bar_t):
    """Compute the correction of EKF

    Parameters:
    x_bar_t       (np.array)    -- the predicted state estimate of time t
    z_t           (np.array)    -- the measured state of time t
    sigma_x_bar_t (np.array)    -- the predicted variance of time t

    Returns:
    x_est_t       (np.array)    -- the filtered state estimate of time t
    sigma_x_est_t (np.array)    -- the filtered variance estimate of time t
    """

    """STUDENT CODE START"""
    H_t = calc_meas_jacobian(x_bar_t)
    K_t = calc_kalman_gain(sigma_x_bar_t, H_t)
    z_bar_t = calc_meas_prediction(x_bar_t)
    x_est_t = x_bar_t + K_t @ (z_t - z_bar_t)
    I = np.identity(5)
    sigma_x_est_t = (I - (K_t @ H_t)) @ sigma_x_bar_t
    """STUDENT CODE END"""

    return [x_est_t, sigma_x_est_t]


def main():
    """Run a EKF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filepath = "../../lab3csv/"
    #filename = '2020_2_26__16_59_7_filtered'
    filename = '2020_2_26__17_21_59_filtered'
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        f_data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    """STUDENT CODE START"""
    N =  5 # number of states
    x_gps, y_gps = convert_gps_to_xy(lat_gps[0],
                                     lon_gps[0],
                                     lat_origin,
                                     lon_origin)
    print('first state', x_gps, y_gps)
    vx_init, vy_init, theta_init = 0, 0, np.radians(0)
    state_est_t_prev = np.array([x_gps, y_gps, vx_init, vy_init, theta_init])
    var_est_t_prev = np.identity(N)

    state_estimates = np.empty((N, len(time_stamps)))
    covariance_estimates = np.empty((N, N, len(time_stamps)))
    gps_estimates = np.empty((2, len(time_stamps)))
    """STUDENT CODE END"""



    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        """STUDENT CODE START"""
        theta_t_prev = wrap_to_pi(getYaw(state_est_t_prev))
        transform_yaw_lidar = wrap_to_pi(np.radians(yaw_lidar[t]))
        dTheta = wrap_to_pi(CCW * transform_yaw_lidar - theta_t_prev)
        omega = dTheta / DT
        u_t = np.array([x_ddot[t], y_ddot[t], omega])
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t, var_pred_t = prediction_step(state_est_t_prev, u_t, var_est_t_prev)

        # Get measurement
        """STUDENT CODE START"""
        theta_t = wrap_to_pi(getYaw(state_pred_t))
        z_t = np.array([ 5 - (y_lidar[t] * np.cos(theta_t) + x_lidar[t] * np.sin(theta_t)),
                        -5 - (y_lidar[t] * np.sin(theta_t) + x_lidar[t] * np.cos(theta_t)),
                        CCW * transform_yaw_lidar])
        """STUDENT CODE END"""

        # Correction Step
        state_est_t, var_est_t = correction_step(state_pred_t,
                                                 z_t,
                                                 var_pred_t)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t
        var_est_t_prev = var_est_t

        # Log Data
        state_estimates[:, t] = state_est_t
        covariance_estimates[:, :, t] = var_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])

    """STUDENT CODE START"""
    xStates = state_estimates[0]
    yStates = state_estimates[1]
    # plt.scatter(xStates[0], yStates[0])
    # plt.scatter(gps_estimates[0][0], gps_estimates[1][0])
    plt.scatter(xStates, yStates, c=np.arange(len(xStates)), cmap="gist_rainbow")
    plt.scatter(gps_estimates[0], gps_estimates[1], c = np.arange(gps_estimates.shape[1]))

    plt.show()
    """STUDENT CODE END"""
    return 0

def visual():
    filepath = "../../lab3csv/"
    filename = '2020_2_26__16_59_7_filtered'
    # filename = '2020_2_26__17_21_59'
    data, is_filtered = load_data(filepath + filename)

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    # plt.plot(x_lidar, label='x')
    # plt.plot(y_lidar, label='y')
    # plt.plot(np.radians(yaw_lidar)-np.pi)
    plt.plot(yaw_lidar)
    # plt.plot(y_ddot)
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
