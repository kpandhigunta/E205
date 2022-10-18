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
from globals import NUM_STATES, NUM_PARTICLES, SEED
from prediction import prediction_step, init_particles
from resample import resample_step
from RMS import find_RMS_error
import progressbar

def moving_average(x, window = 10):
    return np.convolve(x, 1.0 * np.ones(window) / window, 'full')


def main(filename, isInitKnown, isKidnapped):
    """Run a PF on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    shelvename="./shelve.out"
    my_shelf = shelve.open(shelvename) # 'n' for new

    filepath = "../../lab3csv/"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        data = filter_data(data)
        save_data(data, filepath+filename+"_filtered.csv")
    
    # Kidnap robot: stop at timestep 50, restart logging at timestep 350
    if isKidnapped:
        STOP_TIME = 50
        RESTART_TIME = 350
        for key in data:
            data[key] = data[key][:STOP_TIME] + data[key][RESTART_TIME:]

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
    
    np.random.seed(SEED)
    state_est_t_prev = init_particles(
        x_orig=0,
        y_orig=0,
        is_init_known=isInitKnown
    )

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




    """STUDENT CODE START"""
    #plt.figure()
    x_est = []
    y_est = []
    for t, _ in enumerate(time_stamps):
        x = np.average(state_estimates[0,:,t], weights = state_estimates[5,:,t])
        y = np.average(state_estimates[1,:,t], weights = state_estimates[5,:,t])
        x_est.append(x)
        y_est.append(y)
        if t % 100 == 1:
            plt.scatter(state_estimates[0,:,t], state_estimates[1,:,t], color=(0.1, 0.2 + t/780.0/1.8, 0.5, 0.2 + t/780.0/1.8))

    key = str((filename, isInitKnown, isKidnapped))
    try:
        my_shelf[key] = (x_est, y_est)
    except Exception:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

    plt.axis('equal')
    GPS_N = len(gps_estimates[0])
    plt.plot(gps_estimates[0], gps_estimates[1], c='darkblue', label='GPS data')
    plt.plot(x_est, y_est, c='red', label='estimated path')

    # plot perfect square
    plt.plot(np.linspace(0, 10, 100), np.linspace(0, 0, 100), c = 'darkorange')
    plt.plot(np.linspace(10, 10, 100), np.linspace(0, -10, 100), c = 'darkorange')
    plt.plot(np.linspace(0, 10, 100), np.linspace(-10, -10, 100), c = 'darkorange')
    plt.plot(np.linspace(0, 0, 100), np.linspace(-10, 0, 100), c = 'darkorange', label='expected path')

    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.tight_layout()
    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')

    plt.figure()
    print('approximate RMS:', find_RMS_error(x_est, y_est))
    plt.xlabel('time [s]')
    plt.ylabel('error [m]')
    plt.title('Error of estimated path from perfect square')

    # print('approximate RMS:', find_RMS_error(time_stamps, state_estimates[0][:GPS_N], state_estimates[1][:GPS_N]))
    plt.show()
    """STUDENT CODE END"""
    return 0

def performance(filename):
    shelvename = "./shelve.out"
    keyKnown = str((filename, True, False))
    keyUnknown = str((filename, False, False))
    with shelve.open(shelvename) as db:
        x_known, y_known = db[keyKnown]
        x_unknown, y_unknown = db[keyUnknown]
        find_RMS_error(x_known, y_known)
        find_RMS_error(x_unknown, y_unknown)
        plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        plt.show()

if __name__ == "__main__":
    FILEONE = "2020_2_26__17_21_59_filtered"
    FILETWO = "2020_2_26__16_59_7_filtered"

    def runAll():
        main(filename=FILEONE, isInitKnown=True, isKidnapped=False)
        main(filename=FILEONE, isInitKnown=False, isKidnapped=False)
        main(filename=FILETWO, isInitKnown=True, isKidnapped=False)
        main(filename=FILETWO, isInitKnown=False, isKidnapped=False)
    
    # runAll()

    # main(
    #     filename=FILETWO,
    #     isInitKnown=True,
    #     isKidnapped=False
    # )
    performance(FILEONE)
