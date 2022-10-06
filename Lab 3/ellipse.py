import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def printPearsons(state_estimates, covariance_estimates):
    cov_xys = covariance_estimates[1,0,:]
    sigma_xs = np.sqrt(covariance_estimates[0,0,:])
    sigma_ys = np.sqrt(covariance_estimates[1,1,:])
    x_states = state_estimates[0,:]
    y_states = state_estimates[1,:]
    n = 1

    pearsons = cov_xys / sigma_xs / sigma_ys
    print('Smallest Pearsons: ', np.min(pearsons))
    print('Largest Pearsons: ', np.max(pearsons))

    PLOT_INCREMENTS = 80
    DEGREES = 45
    NUM_PLOTS = covariance_estimates.shape[2] // PLOT_INCREMENTS + 1
    
    fig, ax = plt.subplots()
    ax.plot(x_states, y_states, c='red', label='XY Position State Estimates')
    
    for j, i in enumerate(PLOT_INCREMENTS * np.arange(NUM_PLOTS)):
        x = x_states[i]
        y = y_states[i]
        p = pearsons[i]
        horizontal_diameter = 2 * np.sqrt(1 + p)
        vertical_diameter = 2 * np.sqrt(1 - p)
        colors = [
            'lime',
            'orange',
            'green',
            'blue',
            'blueviolet',
            'violet',
            'cyan',
            'maroon',
            'teal',
            'dimgrey'
        ]
        width = 2 * n * sigma_xs[i] * np.cos(np.radians(DEGREES)) * horizontal_diameter
        height = 2 * n * sigma_ys[i] * np.sin(np.radians(DEGREES)) * vertical_diameter
        ax.add_patch(Ellipse((x, y), width, height, DEGREES,fill=False, label='State '+str(i), color=colors[j]))
        ax.set_xlim([-2,12])
        ax.set_ylim([-12,2])
        ax.legend(bbox_to_anchor=(0,1,1,0), loc="lower left", mode="expand", ncol=3)
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        plt.tight_layout()
