# libs
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as cols
from numba import njit, int32, typed
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
    def __eq__(self, other): # two coords are equal if they have the same coordinates (ignore acid)
        if self.x == other.x:
            if self.y == other.y:
                return True
        return False
    def move_to_indices(self, i, j):
        self.i = i
        self.j = j
        [self.x, self.y] = spatial_coord(i, j, self.dim)
    def move_to_spatial(self, x, y):
        self.x = x
        self.y = y
        [self.i, self.j] = index_coord(x, y, self.dim)

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
    if true_count == 0:
        return new_i, new_j

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
    if true_count == 0:
        return new_i, new_j

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
    coord_vec = typed.List(coord_vec)             
    
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
    coord_vec = typed.List(coord_vec)       
    
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
    coord_vec = typed.List(coord_vec)             
    
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

def plot_grid(coord_vec, dim, path):
    fig, ax = plt.subplots()
    for i in range(len(coord_vec)):
        ax.add_artist(plt.Circle((coord_vec[i].x, coord_vec[i].y), 0.1, color = "black"))
        if i > 0:
            ax.plot([coord_vec[i].x, coord_vec[i-1].x], [coord_vec[i].y, coord_vec[i-1].y], color="black")
    ax.set_xlim([-dim - 1, dim + 1])
    ax.set_ylim([-dim - 1, dim + 1])
    ax.set_title(f"{len(coord_vec)} steps")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect('equal')
    fig.savefig(path)
    return fig, ax

def plot_protein(coord_vec, dim, path):
    fig, ax = plt.subplots()
    for i in range(len(coord_vec)):
        if i > 0:
            ax.plot([coord_vec[i].x, coord_vec[i-1].x], [coord_vec[i].y, coord_vec[i-1].y], color="black")
    for i in range(len(coord_vec)):
        ax.add_artist(plt.Circle((coord_vec[i].x, coord_vec[i].y), 0.3, color = cmap[coord_vec[i].amin - 1]))

    ax.set_xlim([-dim - 1, dim + 1])
    ax.set_ylim([-dim - 1, dim + 1])
    ax.set_title(f"{len(coord_vec)} peptids")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect('equal')
    fig.colorbar(ScalarMappable(norm = cols.Normalize(1, 20), cmap=cols.LinearSegmentedColormap.from_list("a", cmap, 20)), 
                                ax=ax, label="Amino acid", ticks = [i for i in range(1, 21)])
    fig.savefig(path)
    return fig, ax

cmap = ["#6C00E6",
        "#A300E4",
        "#D800E1",
        "#DF00B3",
        "#DD007E",
        "#DA004A",
        "#D80017",
        "#D60000",
        "#D30900",
        "#D13A00",
        "#CF6900",
        "#CC9600",
        "#CAC200",
        "#A4C800",
        "#77C500",
        "#4DC300",
        "#24C100",
        "#00BE00",
        "#00BC18",
        "#00BA40",
        "#00B766"]