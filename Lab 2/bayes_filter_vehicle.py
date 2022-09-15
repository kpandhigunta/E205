"""
Author: Sean & Kaanthi
Date of Creation: 9/15/2022
Description:
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_excel(filename):
    """
    Pull vehicle data from excel sheet
    """
    data = pd.read_excel(filename)
    arr = np.array(data)
    return arr

def speed_hist(data):
    X4 = data[:,13]
    Y4 = data[:,14]

    def getPrevNext(vec):
        # first 12 values are NaNs
        return vec[12:-1], vec[13:]

    prevX4, nextX4 = getPrevNext(X4)
    prevY4, nextY4 = getPrevNext(Y4)
    dist = np.sqrt(np.square(nextX4-prevX4)+np.square(nextY4-prevY4))
    print('dist', dist)
    dt = 0.5
    s = dist/dt
    plt.hist(s)
    plt.show()


    

    # for x, i in enumerate(X4):
    #     localx = X_ego[i]
    #     localy = Y_ego[i]
    #     y = Y4[i]
    #     t = time[i]
    #     if x4 != NaN:
    #         distance = np.sqrt(x-)
    return 0


if __name__=='__main__':
    excel_data = load_excel("assets/E205_Lab2_NuScenesData.xlsx")
    print(speed_hist(excel_data))

    