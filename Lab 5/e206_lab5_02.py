from motion_model import run_motion_model
import time
from traj_planner_utils import *
from traj_tracker_02 import *

  
pi = 3.141529

def main():

  # Construct an environmnt

  for trial in range(0, 6):
    initial_state, waypoints, objects, walls = create_motion_planning_problem(trial=trial)

    # Create a motion planning problem and solve it
    desired_traj = waypoints
    
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
    waypoints = [[3, 3, 0, 0]]
  elif trial == 1:
    waypoints = [[3, 3, 0, pi/2]]
  elif trial == 2:
    waypoints = [[3, 0, 3, 0]]
  elif trial == 3:
    waypoints = [[3, 0, 3, pi]]
  elif trial == 4:
    waypoints = [[3, -3, 3, pi]]
  elif trial == 5:
    waypoints = [[3, -3, 0, pi]]
  else:
    waypoints = [[]]

  maxR = 5
  walls = [[-maxR, maxR, maxR, maxR, 2*maxR], [maxR, maxR, maxR, -maxR, 2*maxR], [maxR, -maxR, -maxR, -maxR, 2*maxR], [-maxR, -maxR, -maxR, maxR, 2*maxR] ]
  objects = []
  
  return current_state, waypoints, objects, walls

if __name__ == '__main__':
    main()
    
    
