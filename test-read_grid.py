import sys
import numpy as np
from collections import namedtuple

gridfile = sys.argv[1]
with open(gridfile, "rb") as f:
    grid_data = f.read()


from read_grid import read_grid

grid = read_grid(grid_data)
neighbour_grid = grid.neighbour_grid
print(sorted([a for a in dir(grid) if not a.startswith("_")]))