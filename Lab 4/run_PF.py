"""
Author: Andrew Q. Pham, Victor Shia
Email: apham@g.hmc.edu, vshia@g.hmc.edu
Date of Creation: 2/26/20
Description:
    Particle filter implementation to filtering localization estimate
    This code is for teaching purposes for HMC ENGR205 System Simulation Lab 4
    Student code version with parts omitted.
"""


import matplotlib.pyplot as plt
import numpy as np
import shelve
from utils import *
from prediction import prediction_step
from resample import resample_step


def moving_average(x, window = 10):
    return np.convolve(x, 1.0 * np.ones(window) / window, 'full')


def main():
    """Run a PF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    filename="./shelve.out"
    my_shelf = shelve.open(filename, "n") # 'n' for new

    filepath = "../"
    filename = "2020_2_26__16_59_7_filtered"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(data, filepath+filename+"_filtered.csv")

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

    x_ddot = moving_average(x_ddot)
    y_ddot = moving_average(y_ddot)

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #  Initialize filter
    """STUDENT CODE START"""
    """STUDENT CODE END"""

    #  Run filter over data
    for t, _ in enumerate(time_stamps):
        # Get control input
        """STUDENT CODE START"""
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t = prediction_step(state_est_t_prev, u_t, z_t)
        state_est_t = resample_step(state_pred_t)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        state_est_t_prev = state_est_t

        # Log Data
        state_estimates[:, :, t] = state_est_t

        x_gps, y_gps = convert_gps_to_xy(lat_gps=lat_gps[t],
                                         lon_gps=lon_gps[t],
                                         lat_origin=lat_origin,
                                         lon_origin=lon_origin)
        gps_estimates[:, t] = np.array([x_gps, y_gps])
        lidar_pos[:,t] = np.array([z_x, z_y])

    for key in dir():
        try:
            my_shelf[key] = eval(key)
        except Exception:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


    """STUDENT CODE START"""
    # Plot here
    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()
