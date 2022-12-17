# libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
import matplotlib.colors as cols
from numpy.random import normal, randint, rand
from numba import njit, prange, typed
from copy import deepcopy

# code
import randomwalk

# generate random exchange matrix elements with exp. value 3 and stddev 1/sqrt(2)
@njit
def random_exchange_matrix():
    mu = -3.0
    sig = 1.0/np.sqrt(2)
    # generate non-symmetrical matrix following requested distribution
    J_non_symm = normal(mu, sig, (20, 20))
    # use the fact that the sum of two normally distributed random numbers is normally distributed
    # with mu = mu_1 + mu_2 and sig^2 = sig_1^2 + sig_2^2
    # (here mu_1 = mu_2 so we divide by two to recover the original distribution)
    J_symm = (J_non_symm + J_non_symm.T) / 2
    return J_symm


# Generate a random protein of given length and evolve it mc_steps-times using the given monte carlo scheme
# at temperature T, saving a plot of the total energy at path
def evolve_protein_plot_energy(length, mc_steps, T, path):
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    while coord_vec[-1].x == 0: # discard the protein and re-generate if it doesn't have full length
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_init_l_{length}_steps_{mc_steps}.pdf") # plot initial state

    J = random_exchange_matrix() # generate a random exchange matrix
    ergs = np.empty(mc_steps, dtype=np.double) # save energy at each step
    for k in range(mc_steps):
        grid, coord_vec = monte_carlo_step(grid, coord_vec, J, T) # perform mc steps
        ergs[k] = total_erg_per_site(grid, coord_vec, J)
    randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_final_l_{length}_steps_{mc_steps}.pdf") # plot final state

    fig, ax = plt.subplots()
    ax.plot(np.asarray(range(mc_steps)), ergs, label=f"$L={length}$") # plot energy
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Total energy $E$")
    ax.legend()
    ax.set_title(f"{mc_steps} monte carlo steps")
    fig.savefig(path+f"/energy_l_{length}_steps_{mc_steps}.pdf")
    
    return fig, ax, ergs, grid, coord_vec


# Perform a mc step on the grid, coor_vec pair as explained on the exercise sheet at given temperature T.
@njit
def monte_carlo_step(grid, coord_vec, J, T):
    protein_length = len(coord_vec)
    m = randint(0, protein_length) # select random peptide
    i, j = coord_vec[m].i, coord_vec[m].j # grid coords of the selected peptide

    # check the validity of all theoretically possible folds and perform the valid one (depending on delta E, see below)
    new_grid, new_coord_vec = check_and_perform_fold(grid, coord_vec, J, T, m, i, j,  1,  1) # up-right
    new_grid, new_coord_vec = check_and_perform_fold(grid, coord_vec, J, T, m, i, j,  1, -1) # down-right
    new_grid, new_coord_vec = check_and_perform_fold(grid, coord_vec, J, T, m, i, j, -1,  1) # up-left
    new_grid, new_coord_vec = check_and_perform_fold(grid, coord_vec, J, T, m, i, j, -1, -1) # down-left

    return new_grid, new_coord_vec


# Check if a fold is valid, calculate Delta E and return folded/original protein .
# Delta i, j must be (1, 1), (1, -1), (-1, 1) or (-1, -1), since the folds move the peptide diagonally.
@njit
def check_and_perform_fold(grid, coord_vec, J, T, m, i, j, delta_i, delta_j):
    if check_fold_validity(grid, coord_vec, m, i, j,  delta_i,  delta_j): # check fold, see below
        new_grid = copy_grid(grid) # make a new copied grid and perform the fold on it
        new_grid[i  ,   j] = 0
        new_grid[i+delta_i, j+delta_j] = grid[i, j]
        new_coord_vec = copy_coord_vec(coord_vec) # same spiel for the coordinate vector
        new_coord_vec[m].move_to_indices(i+delta_i, j+delta_j)
        delta_E = +(local_erg(new_grid, new_coord_vec, m, J) - local_erg(grid, coord_vec, m, J)) # +- ???
        if delta_E <= 0: # negative energy change: keep change
            return new_grid, new_coord_vec
        elif rand() < np.exp(-delta_E/T): # positive energy change: keep change only at a certain chance
            return new_grid, new_coord_vec

    return grid, coord_vec


# this does pretty much exactly what it says
@njit
def copy_grid(old):
    dim = len(old[0])
    new = np.zeros((dim, dim), dtype=np.int32)
    for i in range(dim):
        for j in range(dim):
            new[i][j] = old[i][j]
    return new


# this does pretty much exactly what it says
@njit
def copy_coord_vec(old):
    new = [randomwalk.coord(0, 0, 0, 0)] * len(old)
    new = typed.List(new) 
    for m in range(len(old)):
        cp = old[m]
        new[m] = randomwalk.coord(cp.x, cp.y, cp.amin, cp.dim)
    return new


# check the validity of a certain fold and return truth value
# delta i, j as above must correspond to a diagonal shift
@njit
def check_fold_validity(grid, coord_vec, m, i, j, delta_i, delta_j):
    if i+delta_i < 0 or i+delta_i >= len(grid[0]) or j+delta_j < 0 or j+delta_j >= len(grid[0]):
        return False
    if m == 0: # special case: first peptide
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m+1].i, coord_vec[m+1].j] == [i+delta_i, j]:
            return True
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m+1].i, coord_vec[m+1].j] == [i, j+delta_j]:
            return True
    elif m == len(coord_vec)-1: # special case: last peptide
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m-1].i, coord_vec[m-1].j] == [i, j+delta_j]:
            return True
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m-1].i, coord_vec[m-1].j] == [i+delta_i, j]:
            return True
    else: # check if the chain is not broken by the fold and the translated-to space is not occupied
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m-1].i, coord_vec[m-1].j] == [i, j+delta_j] \
                and [coord_vec[m+1].i, coord_vec[m+1].j] == [i+delta_i, j]:
            return True
        if grid[i+delta_i][j+delta_j] == 0  and [coord_vec[m+1].i, coord_vec[m+1].j] == [i, j+delta_j] \
                and [coord_vec[m-1].i, coord_vec[m-1].j] == [i+delta_i, j]:
            return True
    return False        


# Calculate total energy per site of a protein configuration
@njit
def total_erg_per_site(grid, coord_vec, J):
    E = 0
    for m in range(len(coord_vec)):
        # sum local ergs to get total erg. this counts every contribution twice so we divide by two below
        E += local_erg(grid, coord_vec, m, J) 
    return E/(2*len(coord_vec)) # per site!


# calculate the local energy at a peptide side m, i.e. the energy caused by neighbour interactions
@njit
def local_erg(grid, coord_vec, m, J):
    E = 0
    if 0 < m < len(coord_vec)-1:
        current = coord_vec[m] 
        behind  = coord_vec[m-1] # get direct neighbours on the chain
        front   = coord_vec[m+1] # which don't contribute
        i, j = current.i, current.j    # current grid indices
        upper_neighbour = grid[i][j+1] # neighbour peptide numbers
        right_neighbour = grid[i+1][j]
        lower_neighbour = grid[i][j-1]
        left_neighbour  = grid[i-1][j]
        # here: check if neighbours are occupied and not direct chain neighbours. if so: add matrix element to E
        if upper_neighbour != 0 and [i, j+1] != [front.i, front.j] and [i, j+1] != [behind.i, behind.j]:
            E += J[upper_neighbour - 1][current.amin - 1]
        if right_neighbour != 0 and [i+1, j] != [front.i, front.j] and [i+1, j] != [behind.i, behind.j]:
            E += J[right_neighbour - 1][current.amin - 1]
        if lower_neighbour != 0 and [i, j-1] != [front.i, front.j] and [i, j-1] != [behind.i, behind.j]:
            E += J[lower_neighbour - 1][current.amin - 1]
        if left_neighbour  != 0 and [i-1, j] != [front.i, front.j] and [i-1, j] != [behind.i, behind.j]:
            E += J[left_neighbour - 1][current.amin - 1]
    elif m == 0: # special case: m is first peptide
        current = coord_vec[m]        
        front   = coord_vec[m+1]
        i, j = current.i, current.j
        upper_neighbour = grid[i][j+1]
        right_neighbour = grid[i+1][j]
        lower_neighbour = grid[i][j-1]
        left_neighbour  = grid[i-1][j]
        if upper_neighbour != 0 and [i, j+1] != [front.i, front.j]:
            E += J[upper_neighbour - 1][current.amin - 1]
        if right_neighbour != 0 and [i+1, j] != [front.i, front.j]:
            E += J[right_neighbour - 1][current.amin - 1]
        if lower_neighbour != 0 and [i, j-1] != [front.i, front.j]:
            E += J[lower_neighbour - 1][current.amin - 1]
        if left_neighbour  != 0 and [i-1, j] != [front.i, front.j]:
            E += J[left_neighbour - 1][current.amin - 1]
    elif m == len(coord_vec)-1: # special case: m is last peptide
        current = coord_vec[m]
        behind  = coord_vec[m-1]
        i, j = current.i, current.j
        upper_neighbour = grid[i][j+1]
        right_neighbour = grid[i+1][j]
        lower_neighbour = grid[i][j-1]
        left_neighbour  = grid[i-1][j]
        if upper_neighbour != 0 and [i, j+1] != [behind.i, behind.j]:
            E += J[upper_neighbour - 1][current.amin - 1]
        if right_neighbour != 0 and [i+1, j] != [behind.i, behind.j]:
            E += J[right_neighbour - 1][current.amin - 1]
        if lower_neighbour != 0 and [i, j-1] != [behind.i, behind.j]:
            E += J[lower_neighbour - 1][current.amin - 1]
        if left_neighbour  != 0 and [i-1, j] != [behind.i, behind.j]:
            E += J[left_neighbour - 1][current.amin - 1]
    return E

# helper function for eigenvalue_statistics to profit from numba
@njit(parallel=True)
def parallel_ev_calc(runs, eigenvalues):
    for k in prange(runs):
        # generate a random exchange matrix runs-times
        J = random_exchange_matrix()
        ev, discard = np.linalg.eigh(J) # we don't need the eigenvectors! this should use LAPACK or MKL btw :^) thanks numba
        for i in range(k*20, (k+1)*20):
            eigenvalues[i] = ev[i % 20] # store evs


def eigenvalue_statistics(runs, path):
    eigenvalues = np.empty(runs*20, dtype=np.double)
    parallel_ev_calc(runs, eigenvalues)

    ev_mean = np.mean(eigenvalues)
    ev_std = np.std(eigenvalues)
    
    # make a nice picture
    fig, ax = plt.subplots()
    ax.hist(eigenvalues, density=True, bins=1000, label="Eigenvalues")
    ax.vlines(ev_mean, 0, 1, label=f"mean = {np.round(ev_mean, 3)}", color="red")
    ax.vlines([ev_mean-ev_std, ev_mean+ev_std], 0, 1, label = f"1$\sigma = {np.round(ev_std, 1)}$-width", color="green")
    ax.set_ylabel("Eigenvalue occurence $N_\lambda$")
    ax.set_xlabel("Eigenvalue $\lambda$")
    ax.legend()
    ax.set_ylim([0, 0.2])
    ax.set_title(f"{runs} runs")

    if path != "": # empty path: don't save
        fig.savefig(path)


    return fig, ax, eigenvalues, ev_mean, ev_std # return everything!!1!


def animated_monte_carlo(length, mc_steps, T, frame_interval, fps, path):
    num_frames = int(mc_steps/frame_interval)
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    while coord_vec[-1].x == 0: # discard the protein and re-generate if it doesn't have full length
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)

    J = random_exchange_matrix() # generate a random exchange matrix
    ergs = np.empty(mc_steps, dtype=np.double) # save energy at each step
    grids_vecs = [[grid, coord_vec]]
    for k in range(mc_steps):
        grid, coord_vec = monte_carlo_step(grid, coord_vec, J, T) # perform mc steps
        if k % frame_interval == 0:
            grids_vecs.append([grid, coord_vec])
        ergs[k] = total_erg_per_site(grid, coord_vec, J)

    print("MC done, saving gif. This may take a while.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
    for i in range(len(grids_vecs[0][1])):
        if i > 0:
            ax1.plot([grids_vecs[0][1][i].x, grids_vecs[0][1][i-1].x], [grids_vecs[0][1][i].y, grids_vecs[0][1][i-1].y], color="black")
    for i in range(len(grids_vecs[0][1])):
        ax1.add_artist(plt.Circle((grids_vecs[0][1][i].x, grids_vecs[0][1][i].y), 0.3, color = randomwalk.cmap[grids_vecs[0][1][i].amin - 1]))
    ax2.plot([0], ergs[0], color="red")
    
    ax1.set_xlim([-length/3, length/3])
    ax1.set_ylim([-length/3, length/3])
    ax1.set_title(f"{len(coord_vec)} peptids, $T={T}$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_aspect('equal')
    ax2.set_xlabel("Time step $t$")
    ax2.set_ylabel("Total energy $E$")
    ax2.set_title(f"{mc_steps} monte carlo steps")
    fig.colorbar(ScalarMappable(norm = cols.Normalize(1, 20), cmap=cols.LinearSegmentedColormap.from_list("a", randomwalk.cmap, 20)), 
                                ax=ax1, label="Amino acid")
    fig.tight_layout()
    
    
    anim = FuncAnimation(fig, anim_update, frames=num_frames, fargs=(frame_interval, fig, ax1, ax2, grids_vecs, ergs, length, T), interval=100, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(path+f"/anim_test.gif", writer=writer)
    
    return


def anim_update(frame, frame_interval, fig, ax1, ax2, grids_vecs, ergs, length, T):
    ax1.clear()
    ax1.set_xlim([-length/3, length/3])
    ax1.set_ylim([-length/3, length/3])
    ax1.set_title(f"{len(grids_vecs[0][1])} peptids, $T={T}$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    tot_frames = len(grids_vecs)
    for i in range(len(grids_vecs[frame][1])):
        if i > 0:
            ax1.plot([grids_vecs[frame][1][i].x, grids_vecs[frame][1][i-1].x], [grids_vecs[frame][1][i].y, 
                grids_vecs[frame][1][i-1].y], color="black")
    for i in range(len(grids_vecs[frame][1])):
        ax1.add_artist(plt.Circle((grids_vecs[frame][1][i].x, grids_vecs[frame][1][i].y), 0.3, color = randomwalk.cmap[grids_vecs[frame][1][i].amin - 1]))
    ax2.plot(range(frame*frame_interval), ergs[:frame*frame_interval], color="red")
    return
    