"""
Author: Sean & Kaanthi
Date of Creation: 9/15/2022
Description:
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_excel(filename):
    """
    Pull vehicle data from excel sheet
    """
    data = pd.read_excel(filename)
    arr = np.array(data)
    return arr

def getstationaryspeed(data):
    """
    calculate 
    """
    X4 = data[:,13]
    Y4 = data[:,14]
    dt = 0.5

    def getPrevNext(vec):
        # first 12 values are NaNs
        return vec[12:-1], vec[13:]

    prevX4, nextX4 = getPrevNext(X4)
    prevY4, nextY4 = getPrevNext(Y4)
    dist = np.sqrt(np.square(nextX4-prevX4)+np.square(nextY4-prevY4))
    s = dist/dt
    return s

def plothistspeed(speeddata):
    n, bins, patches = plt.hist(speeddata)
    mu, sigma = norm.fit(speeddata)
    y = norm.pdf( bins, mu, sigma)
    plt.plot(bins, y, '--', linewidth=2)
    plt.show()
    return sigma

def problem3():
    excel_data = load_excel("assets/E205_Lab2_NuScenesData.xlsx")
    stationary_data = getstationaryspeed(excel_data)
    sigma = plothistspeed(stationary_data)
    print(sigma)

if __name__=='__main__':
    problem3()

    