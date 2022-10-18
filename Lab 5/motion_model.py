import math
from traj_tracker_02 import TIME_STEP_SIZE
from traj_planner_utils import *

DT = TIME_STEP_SIZE

def run_motion_model(state, input):
    """
    Runs the differential drive motion model for a given input
    Odometry model
    """
    w1 = input[0]
    w2 = input[1]

    w = w1 + w2
    v = w1 - w2

    theta = state[2]
    
    x_n = state[0] + math.cos(theta) * v * DT
    y_n = state[1] + math.sin(theta) * v * DT
    theta_n = angle_diff(theta + w * DT)

    return [x_n, y_n, theta_n]
