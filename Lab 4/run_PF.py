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
from prediction import prediction_step, NUM_PARTICLES
from resample import resample_step
from RMS import find_RMS_error
import progressbar

def moving_average(x, window = 10):
    return np.convolve(x, 1.0 * np.ones(window) / window, 'full')


def main():
    """Run a PF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    # filename="./shelve.out"
    # my_shelf = shelve.open(filename, "n") # 'n' for new

    filepath = "../../lab3csv/"
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
    # Preprocess Yaw
    CCW = -1.0
    yaw_lidar = np.radians(CCW * np.array(yaw_lidar)) # flip counterclockwise
    dTheta = np.pad(yaw_lidar,(0,1)) - np.pad(yaw_lidar,(1,0))
    for i, dQ in enumerate(dTheta):
        dTheta[i] = wrap_to_pi(dQ)
    omega = dTheta / DT

    # Initialize particles
    NUM_STATES = 6
    np.random.seed(1423)
    state_est_t_prev = np.empty((NUM_STATES, NUM_PARTICLES))
    state_est_t_prev[0] = 4 * np.random.random(NUM_PARTICLES) - 2
    state_est_t_prev[1] = 4 * np.random.random(NUM_PARTICLES) - 2
    state_est_t_prev[2] = np.zeros(NUM_PARTICLES)
    state_est_t_prev[3] = np.zeros(NUM_PARTICLES)
    state_est_t_prev[4] = 2 * np.pi * np.random.random(NUM_PARTICLES) - np.pi
    state_est_t_prev[5] = np.ones(NUM_PARTICLES)

    # initalize logs
    state_estimates = np.zeros((NUM_STATES, NUM_PARTICLES, len(time_stamps)))
    gps_estimates = np.zeros((2, len(time_stamps)))
    lidar_pos = np.zeros((2, len(time_stamps)))

    """STUDENT CODE END""" 

    #  Run filter over data
    for t in progressbar.progressbar(range(len(time_stamps))):
        # Get control input
        """STUDENT CODE START"""  
        u_t = np.array([
            x_ddot[t],
            omega[t]
        ])

        # convert lidar measurement to z_t
        theta_t = wrap_to_pi(yaw_lidar[t])
        z_t = np.array([ 5 - (y_lidar[t] * np.cos(theta_t) + x_lidar[t] * np.sin(theta_t)),
                        -5 - (y_lidar[t] * np.sin(theta_t) - x_lidar[t] * np.cos(theta_t)),
                        theta_t])
        z_x, z_y, z_theta = z_t
        """STUDENT CODE END"""

        # Prediction Step
        state_pred_t = prediction_step(state_est_t_prev, u_t, z_t)
        state_est_t = resample_step(state_pred_t)
        
        xytheta_indices = np.array([0,1,4])
                    

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
        




    # for key in dir():
    #     try:
    #         my_shelf[key] = eval(key)
    #     except Exception:
    #         #
    #         # __builtins__, my_shelf, and imported modules can not be shelved.
    #         #
    #         print('ERROR shelving: {0}'.format(key))
    # my_shelf.close()


    """STUDENT CODE START"""
    #plt.figure()
    for t, _ in enumerate(time_stamps):
        if t % 100 == 1:
            plt.scatter(state_estimates[0,:,t], state_estimates[1,:,t], color=(0.1, 0.2 + t/780.0/1.8, 0.5, 0.2 + t/780.0/1.8))

    

    plt.axis('equal')
    GPS_N = len(gps_estimates[0])
    plt.plot(gps_estimates[0], gps_estimates[1], c='darkblue', label='GPS data')
    

    #plt.scatter(z_t_log[0][:GPS_N], z_t_log[1][:GPS_N], s=0.5, c='forestgreen', label='measurement (Lidar)')

    # plot perfect square
    # plt.plot(np.linspace(0, 10, 100), np.linspace(0, 0, 100), c = 'orange')
    # plt.plot(np.linspace(10, 10, 100), np.linspace(0, -10, 100), c = 'orange')
    # plt.plot(np.linspace(0, 10, 100), np.linspace(-10, -10, 100), c = 'orange')
    # plt.plot(np.linspace(0, 0, 100), np.linspace(-10, 0, 100), c = 'orange', label='expected path')

    # plt.plot(state_estimates[0][:GPS_N], state_estimates[1][:GPS_N], c='red', label='estimated path state (KF)')

    # plt.xlabel('x position (m)')
    # plt.ylabel('y position (m)')
    # plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    # #plt.ylim((-20,2))
    # #plt.yticks(np.arange(-20, 2, 5))
    # plt.tight_layout()

    # plt.figure()
    # for i in range(len(state_estimates[4])):
    #     state_estimates[4][i] = wrap_to_pi(state_estimates[4][i])
    # plt.plot(np.linspace(0, 70, len(state_estimates[4])), state_estimates[4])
    # plt.xlabel('time (s)')
    # plt.ylabel('yaw angle (rad)')

    # print('approximate RMS:', find_RMS_error(time_stamps, state_estimates[0][:GPS_N], state_estimates[1][:GPS_N]))
    plt.show()
    """STUDENT CODE END"""
    return 0


if __name__ == "__main__":
    main()
