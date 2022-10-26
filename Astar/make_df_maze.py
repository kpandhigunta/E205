from df_maze import Maze

# Maze dimensions (ncols, nrows)
nx, ny = 5, 5
# Maze entry position
ix, iy = 0, 0

maze = Maze(nx, ny, ix, iy)
maze.add_begin_end = True
maze.add_treasure = True
maze.make_maze()

print(maze)
maze.write_svg('maze.svg')
