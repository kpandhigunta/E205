import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def transformGPS(csv_name):
    arr = np.genfromtxt(csv_name, delimiter=',', skip_header=0, dtype=float)
    arr = arr[1:]  # remove nan due to header
    lat = np.deg2rad(arr[:, 5])
    long = np.deg2rad(arr[:, 6])
    lat_mean = np.mean(lat)
    long_mean = np.mean(long)

    r_earth = 6371000 # in m

    x = r_earth*(lat - lat_mean)*np.cos(lat_mean)
    y = r_earth*(long - long_mean)

    # finding covariance of each direction
    print(np.var(x))
    print(np.var(y))

    plt.plot(x, y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# transformGPS('lab1_azimuth_00.csv')


def generate_histogram(csv_name):
    arr = np.genfromtxt(csv_name, delimiter=',', skip_header=0, dtype=float)
    arr = arr[1:] # remove nan due to header
    ranges = arr[:, 0]

    num_bins = 12
    bin_width = (np.max(ranges) - np.min(ranges))/num_bins
    bins = []
    current_bin = min(ranges)
    while current_bin <= max(ranges):
        bins.append(current_bin + bin_width)
        current_bin = current_bin + bin_width

    plt.hist(ranges, bins=bins, ec='k', density=True)
    plt.xlabel("Range (m)")
    plt.ylabel("Histogram Frequency and PDF Likelihood (count)")
    plt.title('Ranges for Azimuth ' + csv_name)


    plt.xticks(bins, rotation=90)
    mu, std = norm.fit(ranges)
    print('std ' + str(std))
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)

    plt.savefig(csv_name[:-3]+".png", dpi=300)
    plt.show()

def find_z_given_x(x):
    mu = 11 + x
    std = 0.0074950183517010875
    point = 9.272
    return norm.pdf(point, mu, std)

def find_x_given_z():
    px = 0.25
    px1 = find_z_given_x(-1.7)
    px2 = find_z_given_x(-1.72)
    px3 = find_z_given_x(-1.74)
    px4 = find_z_given_x(-1.76)
    pz = (px1 + px2 + px3 + px4)*px

    c1 = (px1*px)/pz
    c2 = (px2*px)/pz # largest cond prob so lidar is at x2
    c3 = (px3*px)/pz
    c4 = (px4*px)/pz

    print(c1, c2, c3, c4)

generate_histogram('lab1_azimuth_00.csv')
# find_x_given_z()
