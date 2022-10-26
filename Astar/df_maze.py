# df_maze.py
import random
from typing import List, Set, Tuple


# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.

class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def __str__(self):
        return f"Cell @ ({self.x}, {self.y}). Walls: {self.walls}."

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

        self.add_begin_end = False
        self.add_treasure = False
        self.treasure_x = random.randint(0, self.nx-1)
        self.treasure_y = random.randint(0, self.ny-1)
        self.treasure_x = random.randint(self.nx/2, self.nx-1)
        self.treasure_y = random.randint(self.ny/2, self.ny-1)


    def cell_at(self, x, y) -> Cell:
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]


    def __str__(self):
        """Return a (crude) string representation of the maze."""
        return self.print_path()


    def write_svg(self, filename, path=[], text=""):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 50
        # Height and width of the maze image (excluding padding), in pixels
        height = 500
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'
                  .format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        def add_cell_rect(f, x, y, colour, pad=5):
            print(f'<rect x="{scx*x+pad}" y="{scy*y+pad}" width="{scx-2*pad}"'
                  f' height="{scy-2*pad}" style="fill:{colour}" />', file=f)
        
        def add_cell_dot(f, x, y, colour, pad=5):
            print(f'<circle cx="{scx*(x+0.5)}" cy="{scy*(y+0.5)}" r="{scx/4}"'
                  f' style="fill:{colour}" />', file=f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 5;\n}', file=f)
            print(']]></style>\n</defs>', file=f)
            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighboring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            if text:
                print(f'<text x="-10" y="-10" class="heavy">{text}</text>', file=f)
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)

            for pt in path:
                add_cell_dot(f, pt[0], pt[1], 'blue')
            if self.add_begin_end:
                add_cell_rect(f, self.ix, self.iy, 'green')
            if self.add_treasure:
                add_cell_rect(f, self.treasure_x, self.treasure_y, 'yellow')

            print('</svg>', file=f)


    def find_valid_neighbors(self, cell):
        """Return a list of unvisited neighbors to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbors = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbor = self.cell_at(x2, y2)
                if neighbor.has_all_walls():
                    neighbors.append((direction, neighbor))
        return neighbors

    def get_walls(self, cell: Cell) -> List[Cell]:
        """Return a set of walls."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        walls: List[Tuple[str, Cell]] = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny) and cell.walls[direction]:
                walls.append((
                    direction,
                    self.cell_at(x2, y2)
                ))
        return walls

    def get_neighbors(self, cell: Cell) -> List[Cell]:
        """Return a set of neighbors."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbors: List[Cell] = []
        for direction, (dx, dy) in delta:
            if not cell.walls[direction]:
                neighbors.append(
                    self.cell_at(
                        cell.x + dx,
                        cell.y + dy,
                    )
                )
        return neighbors

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbors = self.find_valid_neighbors(current_cell)

            if not neighbors:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighboring cell and move to it.
            direction, next_cell = random.choice(neighbors)
            current_cell.knock_down_wall(next_cell, direction)

            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1

        for i in range(0, int(n / 20)):
            if i == 0:
                nx, ny = 0, 0
            else:
                nx = random.randint(0, self.nx-1)
                ny = random.randint(0, self.ny-1)
            current_cell = self.cell_at(nx, ny)
            walls = self.get_walls(current_cell)
            if walls:
                print("removing a wall")
                direction, next_cell = random.choice(walls)
                current_cell.knock_down_wall(next_cell, direction)

    def print_path(self, path=[]):
        """Return a (crude) string representation of the maze."""

        path_points = set(
            f"({pt[0]},{pt[1]})"
            for pt in path
        )

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                position = " "
                if x == self.ix and y == self.iy:
                    position = "o"
                elif x == self.treasure_x and y == self.treasure_y:
                    position = "x"
                elif f"({x},{y})" in path_points:
                    position = "."

                if self.maze_map[x][y].walls['E']:
                    maze_row.append(f'{position}|')
                else:
                    maze_row.append(f'{position} ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)
