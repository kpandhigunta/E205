import numpy as np
import matplotlib.pyplot as plt
import math

def square(list):
    return [i ** 2 for i in list]

def find_RMS_error(estimated_x, estimated_y):
    distances = []
    for i in range(len(estimated_x)):
        dist_all = []
        for top in np.linspace(0, 10, 100): # top
            dist = math.dist([estimated_x[i], estimated_y[i]], [top, 0])
            dist_all.append(dist)
        for right in np.linspace(0, -10, 100): # right
            dist = math.dist([estimated_x[i], estimated_y[i]], [10, right])
            dist_all.append(dist)
        for bottom in np.linspace(0, 10, 100): # bottom
            dist = math.dist([estimated_x[i], estimated_y[i]], [bottom, -10])
            dist_all.append(dist)
        for left in np.linspace(-10, 0, 100): # right
            dist = math.dist([estimated_x[i], estimated_y[i]], [0, left])
            dist_all.append(dist)
        distances.append(min(dist_all))
    plt.figure()
    plt.plot(np.linspace(0, 70, len(distances)), distances)
    plt.xlabel('time (s)')
    plt.ylabel('distance between estimated and expected paths (m)')
    return math.sqrt(sum(square(distances))/len(distances))
