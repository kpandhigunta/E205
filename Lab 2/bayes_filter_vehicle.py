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

def getspeed(X, Y):
    """
    calculate 
    """
    def getPrevNext(vec):
        vec = vec[~np.isnan(vec)]
        return vec[:-1], vec[1:]

    dt = 0.5
    prevX, nextX = getPrevNext(X)
    prevY, nextY = getPrevNext(Y)
    dist = np.sqrt(np.square(nextX-prevX)+np.square(nextY-prevY))
    s = dist/dt
    return s

def plothistspeed(speeddata, scaling):
    n, bins, patches = plt.hist(speeddata)
    mu, sigma = norm.fit(speeddata)
    y = norm.pdf( bins, mu, sigma)
    y = y*scaling
    plt.plot(bins, y, '--', linewidth=2)
    
    plt.xlabel('speed s [m/s]')
    plt.ylabel('likelihood [a.u.]')
    plt.xlim((0, 19.5))
    plt.title('P(s_4|x_4)')
    
    
    plt.show()
    return mu, sigma

CAR_EGO_TIME = 0
CAR_1_TIME = 0
CAR_2_TIME = 0
CAR_3_TIME = 0.5
CAR_4_TIME = 6
CAR_5_TIME = 2
CAR_6_TIME = 8.5

def problem3():
    excel_data = load_excel("assets/E205_Lab2_NuScenesData.xlsx")
    speed1 = getspeed(excel_data[:,4], excel_data[:,5])
    speed2 = getspeed(excel_data[:,7], excel_data[:,8])
    speed3 = getspeed(excel_data[:,10], excel_data[:,11])
    speed4 = getspeed(excel_data[:,13], excel_data[:,14])
    speed5 = getspeed(excel_data[:,16], excel_data[:,17])
    speed6 = getspeed(excel_data[:,19], excel_data[:,20])

    speeds = np.concatenate((speed2, speed3, speed5))
    #mustop, sigmastop = plothistspeed(speed4, 2)
    #munot, sigmamove = plothistspeed(speeds, 375)

    plt.figure(figsize=(14,9))
    for i in range(1,7):
        plt.subplot(int('3'+'2'+str(i)))
        bayes_filter(eval('CAR_' + str(i) + '_TIME'), eval('speed' + str(i)))
        plt.title('CAR ' + str(i))
        plt.xlabel('time (s)')
        plt.ylabel('probability that car is stopped')
    plt.tight_layout()
    plt.savefig('carplots.png', dpi=300)
    plt.show()

def prediction_step(bel_x_t_1): 
    p_stop_given_stop = 0.6
    p_not_given_stop = 0.4
    p_stop_given_not = 0.25
    p_not_given_not = 0.75
    belstop, belnot = bel_x_t_1
    bel_x_t_bar = [ p_stop_given_stop * belstop + p_stop_given_not * belnot, 
                    p_not_given_not * belnot + p_not_given_stop * belstop]
    
    return bel_x_t_bar

def correction_step(z_t, bel_x_t_bar):
    SIGMA_STOP = 0.0480106580440431
    SIGMA_NOT = 2.2653421956155033
    MU_STOP = 0.05361051972597967
    MU_NOT = 1.5799809842770214
    p_z_given_stop = norm.pdf(z_t, MU_STOP, SIGMA_STOP)
    p_z_given_not = norm.pdf(z_t, MU_NOT, SIGMA_NOT)
    belstop, belnot = bel_x_t_bar
    belstopnew = p_z_given_stop*belstop
    belnotnew = p_z_given_not*belnot
    total = belstopnew + belnotnew
    bel_x_t = [belstopnew / total, belnotnew / total]
    return bel_x_t

def bayes_filter(time, speed):
    bel_x_t_1 = [0.5, 0.5]
    bel_x_t_arr = []
    for z_t in speed:
        bel_x_t_bar = prediction_step(bel_x_t_1)
        bel_x_t = correction_step(z_t, bel_x_t_bar)
        bel_x_t_arr.append(bel_x_t[0])
    plt.plot(np.arange(time, time+0.5*len(bel_x_t_arr), 0.5), bel_x_t_arr)
    

if __name__=='__main__':
    problem3()

    