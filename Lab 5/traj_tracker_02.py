from traj_planner_utils import *

pi = 3.141529
TIME_STEP_SIZE = 0.01 #s
LOOK_AHEAD_TIME = 1.0 #s
MIN_DIST_TO_POINT = 0.1 #m
MIN_ANG_TO_POINT = 0.10 #rad


class TrajectoryTracker():
  """ A class to hold the functionality for tracking trajectories.
      Arguments:
        traj (list of lists): A list of traj points Time, X, Y, Theta (s, m, m, rad).
  """
  current_point_to_track = 0
  traj_tracked = False
  traj = []
  end_point = []

  def __init__(self, traj):
    self.current_point_to_track = 0
    self.traj = traj
    self.end_point = traj[-1]
    self.traj_tracked = False

  def get_traj_point_to_track(self, current_state):
    """ Determine which point of the traj should be tracked at the current time.
        Arguments:
          current_state (list of floats): The current Time, X, Y, Theta (s, m, m, rad).
        Returns:
          desired_state (list of floats: The desired state to track - Time, X, Y, Theta (s, m, m, rad).
    """

    """STUDENT CODE START"""
    desired_state = self.traj[self.current_point_to_track]
    delta_x = desired_state[1] - current_state[1]
    delta_y = desired_state[2] - current_state[2]
    rho = math.sqrt((delta_x)**2+(delta_y)**2)
    if rho < 0.1:
      self.traj_tracked = True;
    """STUDENT CODE END"""
    return self.traj[self.current_point_to_track]

  def print_traj(self):
    """ Print the trajectory points.
    """
    print("Traj:")
    for i in range(len(self.traj)):
        print(i,self.traj[i])

  def is_traj_tracked(self):
    """ Return true if the traj is tracked.
        Returns:
          traj_tracked (boolean): True if traj has been tracked.
    """
    return self.traj_tracked

class PointTracker():
  """ A class to determine actions (motor control signals) for driving a robot to a position.
  """
  def __init__(self):
    pass

  def get_dummy_action(self, x_des, x):
    """ Return a dummy action for now
    """
    action = [0.0, 0.0]
    return action

  def point_tracking_control(self, desired_state, current_state):
    """ Return the motor control signals as actions to drive a robot to a desired configuration
        Arguments:
          desired_state (list of floats): The desired Time, X, Y, Theta (s, m, m, rad).
          current_state (list of floats): The current Time, X, Y, Theta (s, m, m, rad).
    """
    # zero all of action
    # right wheel velocity
    # left wheel velocity
    action = [0.0, 0.0]

    """STUDENT CODE START"""
    delta_x = desired_state[1] - current_state[1]
    delta_y =desired_state[2] - current_state[2]
    # print(delta_x, delta_y)
    rho = math.sqrt((delta_x)**2+(delta_y)**2)
    alpha = wrap_to_pi(-1*current_state[3] + math.atan2(delta_y, delta_x))
    
    sign = 1
    BACKWARDS = abs(alpha) > math.pi / 2
    if (BACKWARDS):
      sign = -1
    theta_des = desired_state[3]
    beta = wrap_to_pi(-1*current_state[3] - alpha + sign * theta_des)

    # prioritize k_rho and k_alpha when far 
    if rho < 5:
      k_rho = 1 # greater than 0 for stability
      k_beta = -1 # less than 0 for stability
      k_alpha = 2 # greater than k_rho for stability
    else:
      k_rho = 15 # greater than 0 for stability
      k_beta = -20 # less than 0 for stability
      k_alpha = 20 # greater than k_rho for stability
    v = k_rho*rho
    w = k_alpha*alpha + k_beta*beta

    # Assume r = 1 and L = 1 so omega_1 = r * phidot / (2L) = phidot / 2
    right_wheel = (1/2)*w + (1/2)*v
    left_wheel = (1/2)*w - (1/2)*v
    action = [right_wheel, left_wheel]
    """STUDENT CODE END"""
    return action
