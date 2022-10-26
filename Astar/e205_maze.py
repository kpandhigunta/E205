from typing import List, Set

from df_maze import Cell, Maze
import math
import random

# Maze dimensions (ncols, nrows)
nx, ny = 14, 14
# Maze entry position
ix, iy = 0, 0
# ix, iy = int(nx/2), int(ny/2)


class Node():

    def __init__(self, state: List[int], parent_node: "Node", depth: int, g_cost: float, h_cost: float):
        self.state = state # [x, y]
        self.parent_node = parent_node
        self.depth = depth
        self.g_cost = g_cost  # Current cost
        self.h_cost = h_cost  # Estimated cost to go
        self.f_cost = self.g_cost + self.h_cost  # Total cost
        
    def manhattan_distance_to_node(self, node):
        return abs(self.state[0] - node.state[0]) + abs(self.state[1] - node.state[1])
    
    def manhattan_distance_to_state(self, state):
        return abs(self.state[0] - state[0]) + abs(self.state[1] - state[1])
        
    def euclidean_distance_to_state(self, state):
        return math.sqrt( (self.state[0] - state[0])**2 + (self.state[1] - state[1])**2 )

    def __str__(self):
        return f"Node @ {self.state}, g_cost: {self.g_cost}, f_cost: {self.f_cost}, parent: {self.parent_node.state if self.parent_node else None}"


class PlannerType:
    BREADTH = "b"
    DEPTH = "d"
    UNIFORM = "u"
    GREEDY = "g"
    A_STAR = "a"

    TYPES = [BREADTH, UNIFORM, A_STAR, GREEDY, DEPTH]


class Planner:

    def __init__(self, maze: Maze, type: PlannerType):
        self.maze = maze
        self.fringe = []
        self.type = type
        # self.initial_state (x,y)
        # self.desired_state

    def create_initial_node(self, state):
        """
        Return the initial node
        """
        # Add code here.
        """STUDENT CODE START"""
        g_cost = 0
        h_cost = 0 # should be the full dist to desired, but we pull it off immediately
        """STUDENT CODE END"""
        return Node(state, None, 0, g_cost, h_cost)

    def construct_traj(self, initial_state: List[int], desired_state: List[int]):
        """
        Return:
            - list of (x,y) positions of the path
            - number of nodes expanded
            - cost of the trajectory (node.g_cost)
        """
        self.fringe = []
        self.initial_state = initial_state
        self.desired_state = desired_state

        initial_node = self.create_initial_node(initial_state)
        self.fringe.append(initial_node)

        goal_node = None
        nodes_expanded = 0

        """STUDENT CODE START"""
        g_cost = initial_node.manhattan_distance_to_state(initial_state)
        h_cost = initial_node.manhattan_distance_to_state(desired_state)
        # goal_node = Node(desired_state, PARENT, DEPTH, g_cost, h_cost)
        while not self.fringe:
            
        """STUDENT CODE END"""

        goal_traj = self.build_traj(goal_node)

        return goal_traj, nodes_expanded, goal_node.g_cost
        
    def add_to_fringe(self, node: Node):
        """
        Add a new node onto the fringe
        """
        """STUDENT CODE START"""
        self.fringe.append(node)
        """STUDENT CODE END"""


    def get_best_node_on_fringe(self):
        """
        Get the best node on the fringe
        """
        """STUDENT CODE START"""
        cost = [node.f_cost for node in self.fringe]
        return self.fringe[cost.index(min(cost))]
        """STUDENT CODE END"""

    def estimate_cost_to_goal(self, state: List[int]):
        """
        Estimate the cost to the goal
        """
        """STUDENT CODE START"""
        return 0
        """STUDENT CODE END"""

    def get_children(self, node_to_expand: Node):
        """
        Get valid next nodes to visit.
        Be sure to not go back to the parent node!
        """
        """STUDENT CODE START"""
        return []
        """STUDENT CODE END"""

    def reached_goal(self, current_node: Node):
        """
        Returns if the goal has been reached
        """
        """STUDENT CODE START"""
        return False
        """STUDENT CODE END"""

    def build_traj(self, node: Node):
        """
        Builds the trajectory from the goal node.
        No need to change this.
        """
        traj = [node.state]
        current_node = node.parent_node
        while current_node:
            traj.append(current_node.state)
            current_node = current_node.parent_node
        traj.reverse()
        return traj
    
if __name__ == '__main__':
    maze = Maze(nx, ny, ix, iy)
    maze.add_begin_end = True
    maze.add_treasure = True
    maze.make_maze()
    print(maze)
    maze.write_svg('maze.svg')

    for type in PlannerType.TYPES:
        planner = Planner(maze, type)
        solution, nodes_expanded, cost = planner.construct_traj([ix, iy], [maze.treasure_x, maze.treasure_y])
        print(maze.print_path(solution))
        maze.write_svg(f'maze_{type}.svg', solution, f"Nodes expanded: {nodes_expanded}. Cost: {cost}")
