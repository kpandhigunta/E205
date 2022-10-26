import dubins
import numpy as np

print(dir(dubins))

def getState(x,y,theta):
    return [x, y, theta]

q0 = getState(0, 0, 0)
q1 = getState(5, 5, np.pi/2)
turnRadius = 0.5
print(dubins.shortest_path(q0,q1,turnRadius))
