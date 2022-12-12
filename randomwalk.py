# libs
import numpy as np
from numpy.random import randint
from numba import njit, prange, int32
from numba.experimental import jitclass

# transform grid coordinates into indices
@njit
def index_coord(x, y, dim):
    return [x + dim, y + dim]

# transform indices into grid coordinates
@njit
def spatial_coord(i, j, dim):
    return [i - dim, j - dim]

# pre-compiled class that contains spatial and index coordinates and the number associated with an amino-acid
# the syntax may confuse you, cf. https://numba.readthedocs.io/en/stable/user/jitclass.html
classspec = [
    ('x',    int32),
    ('y',    int32),
    ('i',    int32),
    ('j',    int32),
    ('amin', int32),
    ('dim', int32)
]
@jitclass(classspec)
class coord(object):
    def __init__(self, x, y, amin, dim):
        self.x = x
        self.y = y
        self.i = index_coord(x, y, dim)[0]
        self.j = index_coord(x, y, dim)[1]
        self.amin = amin
        self.dim = dim

# this function randomly chooses a valid direction to move for the self avoiding walk
# edges and occupied spaces are avoided
# new coordinates are returned with returns of -1 signaling no free valid space
@njit
def self_avoiding_step(dim, i, j, grid):
    step_truth_values = [False, False, False, False] # order: up, right, down, left!
    [x, y] = spatial_coord(i, j, dim) # transform i, j to grid coords
    new_i = -1 # initialize return
    new_j = -1

    # checks: is the step a valid space inside the grid and is it not occupied? if yes: set it to True
    if np.abs(y + 1) <= dim and grid[i][j+1] == 0:
        step_truth_values[0] = True
    if np.abs(x + 1) <= dim and grid[i+1][j] == 0:
        step_truth_values[1] = True
    if np.abs(y - 1) <= dim and grid[i][j-1] == 0:
        step_truth_values[2] = True
    if np.abs(x - 1) <= dim and grid[i-1][j] == 0:
        step_truth_values[3] = True

    # count valid steps
    true_count = 0
    for bool in step_truth_values:
        if bool:
            true_count += 1

    # generate random number between 1 and # of valid steps + 1
    # to choose a step randomly with equal probability
    rand = randint(1, true_count+1)
    step = -1
    cmp = 1
    k = 0
    for bool in step_truth_values:
        if bool == True:
            if rand == cmp:
                step = k
                break
            cmp += 1
        k += 1

    # perform step selected above
    if step == 0:
        new_i = i
        new_j = j + 1
    if step == 1:
        new_i = i + 1
        new_j = j
    if step == 2:
        new_i = i
        new_j = j - 1
    if step == 3:
        new_i = i - 1
        new_j = j

    return new_i, new_j

# this function randomly chooses a valid direction to move for the random walk
# edges are avoided, occupied spaces are NOT!
# new coordinates are returned with returns of -1 signaling no free valid space
@njit
def random_step(dim, i, j, grid):
    step_truth_values = [False, False, False, False] # order: up, right, down, left!
    [x, y] = spatial_coord(i, j, dim) # transform i, j to grid coords
    new_i = -1 # initialize return
    new_j = -1

    # checks: is the step a valid space inside the grid? if yes: set it to True
    if np.abs(y + 1) <= dim:
        step_truth_values[0] = True
    if np.abs(x + 1) <= dim:
        step_truth_values[1] = True
    if np.abs(y - 1) <= dim:
        step_truth_values[2] = True
    if np.abs(x - 1) <= dim:
        step_truth_values[3] = True

    # count valid steps
    true_count = 0
    for bool in step_truth_values:
        if bool:
            true_count += 1

    # generate random number between 1 and # of valid steps + 1
    # to choose a step randomly with equal probability
    rand = randint(1, true_count+1)
    step = -1
    cmp = 1
    k = 0
    for bool in step_truth_values:
        if bool == True:
            if rand == cmp:
                step = k
                break
            cmp += 1
        k += 1

    # perform step selected above
    if step == 0:
        new_i = i
        new_j = j + 1
    if step == 1:
        new_i = i + 1
        new_j = j
    if step == 2:
        new_i = i
        new_j = j - 1
    if step == 3:
        new_i = i - 1
        new_j = j

    return new_i, new_j


# this function generates a (2*dim + 1)x(2*dim + 1)-grid populated with a self avoiding random walk starting at the origin
# where grid spaces contain the number associated with the amino acid at that coordinate
#
# a vector sequentially containing the coords and amino acid numbers is also generated to keep 
# track of the order of the protein
#
# dim is the maximum length dimension such that x in [-dim, dim], y in [-dim, dim]!
# steps is the amount of random walk steps to be taken
@njit
def self_avoiding_walk_protein(dim, steps):
    # init grid
    grid = np.zeros((2*dim+1, 2*dim+1), dtype=np.int32)
    # init vector, the coord(0, 0, 0, 0) object can be used later to check for validity!
    coord_vec = [coord(0, 0, 0, 0)] * steps             
    
    # generate first amino acid at (0, 0)
    coord_vec[0] = coord(0, 0, randint(1, 21), dim)             
    grid[coord_vec[0].i][coord_vec[0].j] = coord_vec[0].amin

    # perform steps
    for step in range(1, steps):
        # current pos = pos of last node
        curr_i = coord_vec[step-1].i
        curr_j = coord_vec[step-1].j
        new_i, new_j = self_avoiding_step(dim, curr_i, curr_j, grid)
        if new_i == -1: # if no valid step available: stop            
            break
        # create new random node at determined pos
        [new_x, new_y] = spatial_coord(new_i, new_j, dim)
        coord_vec[step] = coord(new_x, new_y, randint(1, 21), dim)
        grid[new_i, new_j] = coord_vec[step].amin       

    return grid, coord_vec


# this function generates a (2*dim + 1)x(2*dim + 1)-grid populated with a self avoiding random walk starting at the origin
# where grid spaces contain 1 (occupied) or 0 (unoccupied)
#
# a vector sequentially containing the coords is also generated to keep 
# track of the order of the steps
#
# dim is the maximum length dimension such that x in [-dim, dim], y in [-dim, dim]!
# steps is the amount of random walk steps to be taken
@njit
def self_avoiding_walk(dim, steps):
    # init grid
    grid = np.zeros((2*dim+1, 2*dim+1), dtype=np.int32)
    # init vector, the coord(0, 0, 0, 0) object can be used later to check for validity!
    coord_vec = [coord(0, 0, 0, 0)] * steps             
    
    # generate first node at (0, 0)
    coord_vec[0] = coord(0, 0, 1, dim)             
    grid[coord_vec[0].i][coord_vec[0].j] = coord_vec[0].amin

    # perform steps
    for step in range(1, steps):
        # current pos = pos of last node
        curr_i = coord_vec[step-1].i
        curr_j = coord_vec[step-1].j
        new_i, new_j = self_avoiding_step(dim, curr_i, curr_j, grid)
        if new_i == -1: # if no valid step available: stop            
            break
        # create new random node at determined pos
        [new_x, new_y] = spatial_coord(new_i, new_j, dim)
        coord_vec[step] = coord(new_x, new_y, 1, dim)
        grid[new_i, new_j] = coord_vec[step].amin       

    return grid, coord_vec


# this function generates a (2*dim + 1)x(2*dim + 1)-grid populated with a NON-self-avoiding 
# random walk starting at the origin where grid spaces contain 1 (occupied) or 0 (unoccupied)
#
# a vector sequentially containing the coords is also generated to keep 
# track of the order of the steps
#
# dim is the maximum length dimension such that x in [-dim, dim], y in [-dim, dim]!
# steps is the amount of random walk steps to be taken
@njit
def random_walk(dim, steps):
    # init grid
    grid = np.zeros((2*dim+1, 2*dim+1), dtype=np.int32)
    # init vector, the coord(0, 0, 0, 0) object can be used later to check for validity!
    coord_vec = [coord(0, 0, 0, 0)] * steps             
    
    # generate first node at (0, 0)
    coord_vec[0] = coord(0, 0, 1, dim)             
    grid[coord_vec[0].i][coord_vec[0].j] = coord_vec[0].amin

    # perform steps
    for step in range(1, steps):
        # current pos = pos of last node
        curr_i = coord_vec[step-1].i
        curr_j = coord_vec[step-1].j
        new_i, new_j = random_step(dim, curr_i, curr_j, grid)
        if new_i == -1: # if no valid step available: stop            
            break
        # create new random node at determined pos
        [new_x, new_y] = spatial_coord(new_i, new_j, dim)
        coord_vec[step] = coord(new_x, new_y, 1, dim)
        grid[new_i, new_j] = coord_vec[step].amin       

    return grid, coord_vec


