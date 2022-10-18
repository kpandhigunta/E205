# E205 Motion Planning

# Simple planner
# C Clark

from typing import List
import math
import dubins
import matplotlib.pyplot as plt

DISTANCE_STEP_SIZE = 0.1 #m
COLLISION_INDEX_STEP_SIZE = 5
ROBOT_RADIUS = 0.4 #m
OBSTACLE_RADIUS = 0.4 #m
WALL_TOLERANCE = 0.0001

def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]

    Parameters:
    angle (float)   -- unwrapped angle

    Returns:
    angle (float)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi

    while angle <= -math.pi:
        angle += 2*math.pi
    return angle

def construct_dubins_traj(traj_point_0, traj_point_1, ignore_time: bool = False):
  """ Construc a trajectory in the X-Y space and in the time-X,Y,Theta space.
      Arguments:
        traj_point_0 (list of floats): The trajectory's first trajectory point with time, X, Y, Theta (s, m, m, rad).
        traj_point_1 (list of floats): The trajectory's last trajectory point with time, X, Y, Theta (s, m, m, rad).
      Returns:
        traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
        traj_distance (float): The length of the trajectory (m).
  """
  
  """STUDENT CODE START"""
  traj = []
  traj_distance = 0
  """STUDENT CODE END"""

  return traj, traj_distance

def plot_tree(traj_actual, tree: List, objects, walls):
  """ Plot a trajectory in the X-Y space and in the time-X,Y,Theta space.
      Arguments:
        desired_traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
        actual_traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
        objects (list of lists): A list of stationay object states with X, Y, radius (m, m, m).
        walls (list of lists: A list of walls with corners X1, Y1 and X2, Y2 points, length (m, m, m, m, m).
  """
  fig, axis_array = plt.subplots(1,1)
  x_desired = []
  y_desired = []
  for tp in traj_actual:
    x_desired.append(tp[1])
    y_desired.append(tp[2])
  axis_array.plot(x_desired, y_desired, 'b')
  axis_array.plot(x_desired[0], y_desired[0], 'ko')
  axis_array.plot(x_desired[-1], y_desired[-1], 'kx')

  for node in tree:
    if node.parent_node:
      edgeX = [node.parent_node.state[1], node.state[1]]
      edgeY = [node.parent_node.state[2], node.state[2]]
      axis_array.plot(edgeX, edgeY, 'k')

  ang_res = 0.2
  for o in objects:
    x_obj = []
    y_obj = []
    ang = 0
    while ang < 6.28:
      x_obj.append(o[0]+o[2]*math.cos(ang))
      y_obj.append(o[1]+o[2]*math.sin(ang))
      ang += ang_res
    x_obj.append(x_obj[0])
    y_obj.append(y_obj[0])
    axis_array.plot(x_obj, y_obj, 'b')
  for w in walls:
    axis_array.plot([w[0], w[2]], [w[1], w[3]], 'k')
  axis_array.set_xlabel('X (m)')
  axis_array.set_ylabel('Y (m)')
  axis_array.axis('equal')
  plt.show()

def plot_traj(traj_desired, traj_actual, objects, walls):
  """ Plot a trajectory in the X-Y space and in the time-X,Y,Theta space.
      Arguments:
        desired_traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
        actual_traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
        objects (list of lists): A list of stationay object states with X, Y, radius (m, m, m).
        walls (list of lists: A list of walls with corners X1, Y1 and X2, Y2 points, length (m, m, m, m, m).
  """
  fig, axis_array = plt.subplots(2,1)
  time_stamp_desired = []
  x_desired = []
  y_desired = []
  theta_desired = []
  for tp in traj_desired:
    time_stamp_desired.append(tp[0])
    x_desired.append(tp[1])
    y_desired.append(tp[2])
    theta_desired.append(angle_diff(tp[3]))
  axis_array[0].plot(x_desired, y_desired, 'b')
  axis_array[0].plot(x_desired[0], y_desired[0], 'ko')
  axis_array[0].plot(x_desired[-1], y_desired[-1], 'kx')
  time_stamp_actual = []
  x_actual = []
  y_actual = []
  theta_actual = []
  for tp in traj_actual:
    time_stamp_actual.append(tp[0])
    x_actual.append(tp[1])
    y_actual.append(tp[2])
    theta_actual.append(angle_diff(tp[3]))
  axis_array[0].plot(x_actual, y_actual, 'k')

  ang_res = 0.2
  for o in objects:
    x_obj = []
    y_obj = []
    ang = 0
    while ang < 6.28:
      x_obj.append(o[0]+o[2]*math.cos(ang))
      y_obj.append(o[1]+o[2]*math.sin(ang))
      ang += ang_res
    x_obj.append(x_obj[0])
    y_obj.append(y_obj[0])
    axis_array[0].plot(x_obj, y_obj, 'b')
  for w in walls:
    axis_array[0].plot([w[0], w[2]], [w[1], w[3]], 'k')
  axis_array[0].set_xlabel('X (m)')
  axis_array[0].set_ylabel('Y (m)')
  axis_array[0].axis('equal')
  
  axis_array[1].plot(time_stamp_desired, x_desired,'b')
  axis_array[1].plot(time_stamp_desired, y_desired,'b--')
  axis_array[1].plot(time_stamp_desired, theta_desired,'g-.')
  axis_array[1].plot(time_stamp_actual, x_actual,'k')
  axis_array[1].plot(time_stamp_actual, y_actual,'k--')
  axis_array[1].plot(time_stamp_actual, theta_actual,'r-.')
  axis_array[1].set_xlabel('Time (s)')
  axis_array[1].legend(['X Desired (m)', 'Y Desired (m)', 'Theta Desired (rad)', 'X (m)', 'Y (m)', 'Theta (rad)'])

  plt.show()

def collision_found(traj, objects, walls):
  """ Return true if there is a collision with the traj and the workspace
      Arguments:
        traj (list of lists): A list of traj points - Time, X, Y, Theta (s, m, m, rad).
        objects (list of lists): A list of object states - X, Y, radius (m, m, m).
        walls (list of lists): A list of walls defined by end points - X0, Y0, X1, Y1, length (m, m, m, m, m).
      Returns:
        collision_found (boolean): True if there is a collision.
  """

  def _check_traj(index, prev_index):
    traj_point = traj[index]
    prev_traj_point = traj[prev_index]
    line_segment = [
      prev_traj_point[1],
      prev_traj_point[2],
      traj_point[1],
      traj_point[2],
    ]
    line_segment.append(
      math.sqrt((line_segment[2] - line_segment[0]) ** 2 + (line_segment[3] - line_segment[1]) ** 2)
    )
    for obj in objects:
      if index == prev_index:
        obj_distance = generate_distance_to_object(traj_point, obj) - OBSTACLE_RADIUS - ROBOT_RADIUS
      else:
        obj_distance = generate_distance_to_line_segment_for_obj(obj, line_segment) - OBSTACLE_RADIUS - ROBOT_RADIUS
      if obj_distance < 0:
        # print("Ran into object")
        return True
    for wall in walls:
      if index == prev_index:
        wall_distance = generate_distance_to_wall(traj_point, wall) - ROBOT_RADIUS
        if wall_distance < 0:
          # print(f"Ran into wall - 1: {wall}")
          return True
      else:
        intersect = do_line_segments_intersect(line_segment, wall)
        if intersect:
          # print(f"Ran into wall - 2: {wall}. {line_segment}. {index}. {prev_index}")
          return True
    return False

  index = 0
  prev_index = 0
  while index < len(traj):
    has_collision = _check_traj(index, prev_index)
    if has_collision:
      return True
    prev_index = index
    index += COLLISION_INDEX_STEP_SIZE

  if len(traj) <= 1:
    return False
  return _check_traj(prev_index, len(traj) - 1)
  
def generate_distance_to_object(traj_point, obj):
  """ Calculate the distance between a spherical object and a cylindrical robot.
      Argument:
        traj_point (list of floats): A state of Time, X, Y, Theta (s, m, m, rad).
        obj (list of floats): An object state X, Y, radius (m, m, m).
      Returns:
        distance (float): The distance between a traj point and an object (m).
  """
  return math.sqrt( pow(traj_point[1]-obj[0],2) + pow(traj_point[2]-obj[1],2) )
  
def generate_distance_to_wall(traj_point, wall):
  """ Calculate the distance between a spherical object and a cylindrical robot.
      Argument:
        traj_point (list of floats): A state of Time, X, Y, Theta (s, m, m, rad).
        wall (list of floats): An wall state X0, Y0, X1, Y1, length (m, m, m, m, m).
      Returns:
        distance (float): The distance between a traj point and an object (m).
  """
  x0 = traj_point[1]
  y0 = traj_point[2]
  x1 = wall[0]
  y1 = wall[1]
  x2 = wall[2]
  y2 = wall[3]
  num = 1.0 * abs( (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1) )
  den = wall[4]
  
  return num/den
  
def generate_distance_to_line_segment_for_obj(point, line_segment):
  """ Calculate the distance between a spherical object and a cylindrical robot.
      Argument:
        point (list of floats): A state of X, Y, Theta (m, m, rad).
        line_segment (list of floats): An line_segment state X0, Y0, X1, Y1, length (m, m, m, m, m).
      Returns:
        distance (float): The distance between a traj point and an object (m).
  """
  x0 = point[0]
  y0 = point[1]
  x1 = line_segment[0]
  y1 = line_segment[1]
  x2 = line_segment[2]
  y2 = line_segment[3]
  num = 1.0 * abs( (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1) )
  den = line_segment[4]
  
  return num/den
  
def do_line_segments_intersect(line_segment1, line_segment2):
  """ Calculate the distance between a spherical object and a cylindrical robot.
      Argument:
        point (list of floats): A state of Time, X, Y, Theta (s, m, m, rad).
        line_segment (list of floats): An line_segment state X0, Y0, X1, Y1, length (m, m, m, m, m).
      Returns:
        distance (float): The distance between a traj point and an object (m).
  """

  x1a = line_segment1[0]
  y1a = line_segment1[1]
  x2a = line_segment1[2]
  y2a = line_segment1[3]
  x1b = line_segment2[0]
  y1b = line_segment2[1]
  x2b = line_segment2[2]
  y2b = line_segment2[3]

  A1 = line_segment1[3] - line_segment1[1]
  B1 = line_segment1[0] - line_segment1[2]
  C1 = A1 * line_segment1[0] + B1 * line_segment1[1]
  
  A2 = line_segment2[3] - line_segment2[1]
  B2 = line_segment2[0] - line_segment2[2]
  C2 = A2 * line_segment2[0] + B2 * line_segment2[1]

  det = A1 * B2 - A2 * B1
  if det == 0:
    return False

  x = (B2 * C1 - B1 * C2) / det
  y = (A1 * C2 - A2 * C1) / det

  if min(x1a, x2a) - WALL_TOLERANCE <= x and x <= max(x1a, x2a) + WALL_TOLERANCE and min(y1a, y2a) - WALL_TOLERANCE <= y and y <= max(y1a, y2a) + WALL_TOLERANCE and min(x1b, x2b) - WALL_TOLERANCE <= x and x <= max(x1b, x2b) + WALL_TOLERANCE and min(y1b, y2b) - WALL_TOLERANCE <= y and y <= max(y1b, y2b) + WALL_TOLERANCE:
    return True
  else:
    return False
  
def print_traj(traj):
  """ Print a trajectory as a list of traj points.
      Arguments:
        traj (list of lists): A list of trajectory points with time, X, Y, Theta (s, m, m, rad).
  """
  print("TRAJECTORY")
  for tp in traj:
    print("traj point - time:",tp[0], "x:", tp[1], "y:", tp[2], "theta:", tp[3] )
    
def angle_diff(ang):
  """ Function to push ang within the range of -pi and pi
      Arguments:
        ang (float): An angle (rad).
      Returns:
        ang (float): The angle, but bounded within -pi and pi (rad).
  """
  while ang > math.pi:
    ang -= 2*math.pi
  while ang < -math.pi:
    ang += 2*math.pi

  return ang
  
if __name__ == '__main__':
  tp0 = [0,0,0,0]
  tp1 = [10,4,-4, -1.57]
  traj, distance = construct_dubins_traj(tp0, tp1)
  maxR = 8
  walls = [[-maxR, maxR, maxR, maxR], [maxR, maxR, maxR, -maxR], [maxR, -maxR, -maxR, -maxR], [-maxR, -maxR, -maxR, maxR] ]
  objects = [[4, 0, 1.0], [-2, -3, 1.5]]
  plot_traj(traj, traj, objects, walls)


  line_segment1 = [-2.3472901469111482, -5.131496097885485, -2.2725909672157742, -4.637107553769788, 0.5000000000000004]
  line_segment2 = [-10, -5, 5.0, -5, 15.0]
  print(do_line_segments_intersect(line_segment1, line_segment2))
  print(do_line_segments_intersect(line_segment2, line_segment1))
