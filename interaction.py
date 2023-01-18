# auth: Matthias

# libs
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


# code
import randomwalk
import energy
import barrier

@njit
def local_quadr_erg(grid, coord_vec, m, J):
    '''
        gibt <H^2> an einem Ort der Kette aus
    '''
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
            E += J[upper_neighbour - 1][current.amin - 1]**2
        if right_neighbour != 0 and [i+1, j] != [front.i, front.j] and [i+1, j] != [behind.i, behind.j]:
            E += J[right_neighbour - 1][current.amin - 1]**2
        if lower_neighbour != 0 and [i, j-1] != [front.i, front.j] and [i, j-1] != [behind.i, behind.j]:
            E += J[lower_neighbour - 1][current.amin - 1]**2
        if left_neighbour  != 0 and [i-1, j] != [front.i, front.j] and [i-1, j] != [behind.i, behind.j]:
            E += J[left_neighbour - 1][current.amin - 1]**2
    elif m == 0: # special case: m is first peptide
        current = coord_vec[m]
        front   = coord_vec[m+1]
        i, j = current.i, current.j
        upper_neighbour = grid[i][j+1]
        right_neighbour = grid[i+1][j]
        lower_neighbour = grid[i][j-1]
        left_neighbour  = grid[i-1][j]
        if upper_neighbour != 0 and [i, j+1] != [front.i, front.j]:
            E += J[upper_neighbour - 1][current.amin - 1]**2
        if right_neighbour != 0 and [i+1, j] != [front.i, front.j]:
            E += J[right_neighbour - 1][current.amin - 1]**2
        if lower_neighbour != 0 and [i, j-1] != [front.i, front.j]:
            E += J[lower_neighbour - 1][current.amin - 1]**2
        if left_neighbour  != 0 and [i-1, j] != [front.i, front.j]:
            E += J[left_neighbour - 1][current.amin - 1]**2
    elif m == len(coord_vec)-1: # special case: m is last peptide
        current = coord_vec[m]
        behind  = coord_vec[m-1]
        i, j = current.i, current.j
        upper_neighbour = grid[i][j+1]
        right_neighbour = grid[i+1][j]
        lower_neighbour = grid[i][j-1]
        left_neighbour  = grid[i-1][j]
        if upper_neighbour != 0 and [i, j+1] != [behind.i, behind.j]:
            E += J[upper_neighbour - 1][current.amin - 1]**2
        if right_neighbour != 0 and [i+1, j] != [behind.i, behind.j]:
            E += J[right_neighbour - 1][current.amin - 1]**2
        if lower_neighbour != 0 and [i, j-1] != [behind.i, behind.j]:
            E += J[lower_neighbour - 1][current.amin - 1]**2
        if left_neighbour  != 0 and [i-1, j] != [behind.i, behind.j]:
            E += J[left_neighbour - 1][current.amin - 1]**2
    return E


@njit
def total_erg_quadr(grid, coord_vec, J):
    '''
        gibt <H^2>/N aus
        Calculate total energy squared per site of a protein configuration
    '''
    E = 0
    for m in range(len(coord_vec)):
        # sum local ergs to get total erg. this counts every contribution twice so we divide by two below
        E += local_quadr_erg(grid, coord_vec, m, J)
    return E/(2*len(coord_vec)) # per site!

def spec_heat(grid, coord_vec, J, T):
    return np.power(J[0][0],2)*(total_erg_quadr(grid, coord_vec, J) - energy.total_erg_per_site(grid, coord_vec, J)**2)/T**2

def given_interaction_matrix(length, T_steps, num_at_T, T_i, T_f, J,path):
    '''
    length:      length of protein
    T_steps:     total steps
    num_at_t:    how many steps per temperature
    T_i:         initial temperature
    T_f:         final temperature
    J:           interaction matrix J
    path:        where to save plots
    '''
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    while coord_vec[-1].x == 0: # discard the protein and re-generate if it doesn't have full length
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    if path != "":
        figPrev, axPrev = randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_init_l_{length}_steps_{num_at_T*T_steps}_annealing.pdf") # plot initial state
    else:
        figPrev, axPrev = randomwalk.plot_protein(coord_vec, length/3, "")
    
    fullergs = np.empty(T_steps*num_at_T, np.double)
    ergs = np.empty(T_steps, np.double)
    d_ergs = np.empty(T_steps, np.double)
    heat = np.empty(T_steps, np.double)
    geo_distance = np.empty(T_steps, np.double)

    if T_f == 0:
        T_f = 0.01
    dT = (T_i - T_f)/T_steps
    T = T_i

    for i in range(T_steps):
        ergs_at_T = np.empty(num_at_T, np.double)
        heat_at_T = np.empty(num_at_T, np.double)
        geo_at_T = np.empty(num_at_T, np.double)
        for k in range(num_at_T):
            grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, T) # perform mc steps
            ergs_at_T[k] = energy.total_erg_per_site(grid, coord_vec, J)
            heat_at_T[k] = spec_heat(grid, coord_vec, J, T)

            x_diff = coord_vec[-1].x - coord_vec[0].x
            y_diff = coord_vec[-1].y - coord_vec[0].y
            geo_at_T[k] = np.sqrt(x_diff**2 + y_diff**2)

        ergs[i] = np.mean(ergs_at_T)
        d_ergs[i] = np.std(ergs_at_T)
        heat[i] = np.mean(heat_at_T)
        geo_distance[i] = np.mean(geo_at_T)
        fullergs[i*num_at_T:(i+1)*num_at_T] = ergs_at_T
        T -= dT

    if path != "":
        randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_final_l_{length}_steps_{num_at_T*T_steps}_annealing.pdf") # plot initial state
    
    # TEMP FIG
    Temps = np.linspace(T_i, T_f, T_steps)
    fig, ax = plt.subplots()
    #ax.plot(Temps, ergs, label=f"$L={length}$")
    ax.plot(np.asarray(range(T_steps)), ergs, label=f"$L={length}$")
    ax.set_xlabel("T-Step")
    ax.set_ylabel("Total energy per site $E/N$")
    ax.legend()

    ax2 = ax.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, {num_at_T} steps per T, $J_{'ij'} = -3$")
    if path != "":
        fig.savefig(path+f"/given_J_avg_{num_at_T}_l_{length}_steps_{num_at_T*T_steps}.pdf")
    
    # Heat FIG
    Temps = np.linspace(T_i, T_f, T_steps)
    figC, axC = plt.subplots()
    #ax.plot(Temps, ergs, label=f"$L={length}$")
    axC.plot(np.asarray(range(T_steps)), heat, label=f"$L={length}$")
    axC.semilogy()
    axC.set_xlabel("T-Step")
    axC.set_ylabel("Specific heat per site $C$")
    axC.legend()

    axC2 = axC.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    axC2.set_xlabel("Temperature $T$")

    axC.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, {num_at_T} steps per T")
    if path != "":
        figC.savefig(path+f"/spec_heat_avg_{num_at_T}_l_{length}_steps_{num_at_T*T_steps}.pdf")

    # GEO FIG
    fig_geo, ax_geo = plt.subplots()
    ax_geo.plot(np.asarray(range(T_steps)), geo_distance, label=f"$L={length}$")
    ax_geo.set_xlabel("T-Step")
    ax_geo.set_ylabel("Euclidean distance between first and last amino acid")
    ax_geo.legend()

    ax2_geo = ax_geo.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2_geo.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, {num_at_T} steps per T")
    if path != "":
        fig.savefig(path+f"/geometric_distance_avg_{num_at_T}_l_{length}_steps_{num_at_T*T_steps}.pdf")

    return ergs, grid, coord_vec, fig, ax, ax2, figPrev, axPrev, figC, axC, axC2, fig_geo, ax_geo, ax2_geo


def time_to_metastable(length, mc_steps, T, J, runs):
    times = []

    for run in range(runs):    
        fig, ax, ergs, grid, coord_vec, figPrev, axPrev = energy.evolve_protein_plot_energy_given_J(length, mc_steps, T, J, "")
        plt.close(fig)
        plt.close(figPrev)
        step_first_metastable = barrier.time_to_metastable(ergs)
        if step_first_metastable != -1:
            times.append(step_first_metastable)
    
    time = np.mean(times)
    dtime = np.std(times)/np.sqrt(len(times))

    return time, dtime