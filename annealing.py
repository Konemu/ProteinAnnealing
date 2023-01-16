# libs
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# code
import randomwalk
import energy
import randomwalk_statistics

def averaged_annealing(length, T_steps, num_at_T, T_i, T_f, path):
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    while coord_vec[-1].x == 0: # discard the protein and re-generate if it doesn't have full length
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    if path != "":
        figPrev, axPrev = randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_init_l_{length}_steps_{num_at_T*T_steps}_annealing.pdf") # plot initial state
    else:
        figPrev, axPrev = randomwalk.plot_protein(coord_vec, length/3, "")

    J = energy.random_exchange_matrix()
    
    ergs = np.empty(T_steps, np.double)
    d_ergs = np.empty(T_steps, np.double)
    geo_distance = np.empty(T_steps, np.double)

    if T_f == 0:
        T_f = 0.01
    dT = (T_i - T_f)/T_steps
    T = T_i

    for i in range(T_steps):
        ergs_at_T = np.empty(num_at_T, np.double)
        geo_at_T = np.empty(num_at_T, np.double)

        for k in range(num_at_T):
            grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, T) # perform mc steps
            ergs_at_T[k] = energy.total_erg_per_site(grid, coord_vec, J)

            x_diff = coord_vec[-1].x - coord_vec[0].x
            y_diff = coord_vec[-1].y - coord_vec[0].y
            geo_at_T[k] = np.sqrt(x_diff**2 + y_diff**2)

        ergs[i] = np.mean(ergs_at_T)
        d_ergs[i] = np.std(ergs_at_T)
        T -= dT

        geo_distance[i] = np.mean(geo_at_T)

    if path != "":
        randomwalk.plot_protein(coord_vec, length/3, path+f"/protein_final_l_{length}_steps_{num_at_T*T_steps}_annealing.pdf") # plot initial state
    
    Temps = np.linspace(T_i, T_f, T_steps)
    fig, ax = plt.subplots()
    #ax.plot(Temps, ergs, label=f"$L={length}$")
    ax.plot(np.asarray(range(T_steps)), ergs, label=f"$L={length}$")
    ax.set_xlabel("T-Step")
    ax.set_ylabel("Total energy per site $E/N$")
    ax.semilogx()
    ax.legend()

    ax2 = ax.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2.set_xlabel("Temperature $T$")
    step_ticks = ax.get_xticks()
    T_ticks = (T_i - step_ticks)/dT
    ax2.set_xticks(T_ticks)

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, {num_at_T} steps per T")
    if path != "":
        fig.savefig(path+f"/annealing_energy_avg_{num_at_T}_l_{length}_steps_{num_at_T*T_steps}.pdf")

    fig_geo, ax_geo = plt.subplots()
    ax_geo.plot(np.asarray(range(T_steps)), geo_distance, label=f"$L={length}$")
    ax_geo.set_xlabel("T-Step")
    ax_geo.set_ylabel("Euclidean distance between first and last amino acid")
    ax_geo.semilogx()
    ax_geo.legend()

    ax2_geo = ax_geo.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2_geo.set_xlabel("Temperature $T$")
    step_ticks = ax_geo.get_xticks()
    T_ticks = (T_i - step_ticks)/dT
    ax2_geo.set_xticks(T_ticks)

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, {num_at_T} steps per T")
    if path != "":
        fig.savefig(path+f"/geometric_distance_avg_{num_at_T}_l_{length}_steps_{num_at_T*T_steps}.pdf")

    return ergs, grid, coord_vec, fig, ax, ax2, geo_distance, fig_geo, ax_geo, ax2_geo, figPrev, axPrev
