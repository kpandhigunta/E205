from traj_planner_utils import *
import matplotlib.pyplot as plt


pi = 3.141529

def main():
  fig, ax = plt.subplots(2, 2)

  desired_traj, distances = construct_dubins_traj([0, 0, 0, 0], [5, 3, 3, pi/2])
  t, x, y, theta = zip(*desired_traj)
  print(desired_traj[0])
  print(desired_traj[1])
  ax[0, 0].plot(x[0], y[0], "ro")
  ax[0, 0].plot(x[-1], y[-1], "gx")
  ax[0, 0].plot(x, y)
  ax[0, 0].plot(x, y, "k.")
  ax[0, 0].set_title("Problem 1")

  desired_traj, distances = construct_dubins_traj([0, -3, -3,  -pi/4], [10, 3, 3, -pi/4])
  t, x, y, theta = zip(*desired_traj)
  ax[0, 1].plot(x[0], y[0], "ro")
  ax[0, 1].plot(x[-1], y[-1], "gx")
  ax[0, 1].plot(x, y)
  ax[0, 1].plot(x, y, "k.")
  ax[0, 1].set_title("Problem 2")

  desired_traj, distances = construct_dubins_traj([0, -3, 3, 0], [5, 3, -3, 0])
  t, x, y, theta = zip(*desired_traj)
  ax[1, 0].plot(x[0], y[0], "ro")
  ax[1, 0].plot(x[-1], y[-1], "gx")
  ax[1, 0].plot(x, y)
  ax[1, 0].plot(x, y, "k.")
  ax[1, 0].set_title("Problem 3")

  desired_traj, distances = construct_dubins_traj([0, -3, 0, pi/2], [10, 3, 0, -pi/2])
  t, x, y, theta = zip(*desired_traj)
  ax[1, 1].plot(x[0], y[0], "ro")
  ax[1, 1].plot(x[-1], y[-1], "gx")
  ax[1, 1].plot(x, y)
  ax[1, 1].plot(x, y, "k.")
  ax[1, 1].set_title("Problem 4")

  plt.show()


def create_motion_planning_problem():
  current_state = [0, -3, -3,  -pi/4]
  desired_state = [10, 3, 3, -pi/4]
  maxR = 8
  walls = [[-maxR, maxR, maxR, maxR, 2*maxR], [maxR, maxR, maxR, -maxR, 2*maxR], [maxR, -maxR, -maxR, -maxR, 2*maxR], [-maxR, -maxR, -maxR, maxR, 2*maxR] ]
  objects = [[4, 0, 1.0], [-2, -3, 1.5]]

  return current_state, desired_state, objects, walls

if __name__ == '__main__':
    main()


