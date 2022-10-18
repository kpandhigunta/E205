from motion_model import run_motion_model
import time
from traj_planner_utils import *
from traj_tracker_03 import *


pi = 3.141529

def main():

  # Construct an environmnt

  for trial in range(0, 4):
    initial_state, waypoints, objects, walls = create_motion_planning_problem(trial=trial)

    # Create a motion planning problem and solve it
    start = [0, initial_state[0], initial_state[1], initial_state[2]]
    full_traj = []
    for waypoint in waypoints:
        pts = construct_dubins_traj(start, waypoint)
        full_traj.extend(pts[0])
        start = waypoint
    desired_traj = full_traj

    # Create the trajectory and tracking controller
    controller = PointTracker()
    traj_tracker = TrajectoryTracker(desired_traj)

    # Create the feedback loop
    time.sleep(1)
    current_time_stamp = 0
    observation = initial_state
    actual_traj = []
    while not traj_tracker.is_traj_tracked():
        current_state = [current_time_stamp, observation[0], observation[1], observation[2]]
        desired_state = traj_tracker.get_traj_point_to_track(current_state)
        action = controller.point_tracking_control(desired_state, current_state)
        new_observation = run_motion_model(observation, action)
        observation = new_observation

        actual_traj.append(current_state)
        current_time_stamp += TIME_STEP_SIZE
    time.sleep(2)
    plot_traj(desired_traj, actual_traj, objects, walls)


def create_motion_planning_problem(trial: int = 0):
  current_state = [0, 0, 0]
  if trial == 0:
    waypoints = [[5, 10, 0, 0], [10, 10, 10, pi/2], [15, 0, 10, pi], [20, 0, 0, -pi/2]]
  elif trial == 1:
    waypoints = [[5, 10, 0, pi/2], [10, 10, 10, pi], [15, 0, 10, -pi/2], [20, 0, 0, 0]]
  elif trial == 2:
    waypoints = [[5, 10, 0, 0], [10, 10, 2, pi], [15, 0, 2, pi], [20, 0, 4, 0], [25, 10, 4, 0]]
  elif trial == 3:
    waypoints = [[5, 6, 6, pi/2], [10, 0, 12, pi], [15, -6, 6, -pi/2], [20, 0, 0, 0]]
  else:
    waypoints = [[]]

  maxR = 20
  walls = [[-maxR, maxR, maxR, maxR, 2*maxR], [maxR, maxR, maxR, -maxR, 2*maxR], [maxR, -maxR, -maxR, -maxR, 2*maxR], [-maxR, -maxR, -maxR, maxR, 2*maxR] ]
  objects = []

  return current_state, waypoints, objects, walls

if __name__ == '__main__':
    main()
