# libs
import numpy as np
import matplotlib.pyplot as plt

# code
import randomwalk
import energy

def annealing_multiple_runs(length, T_steps, num_at_T, runs, T_i, T_f, path):
    delta_ergs_avg = np.zeros(T_steps, np.double)
    geo_dist_avg = np.zeros(T_steps, np.double)

    for runIndex in range(runs):
        # create protein on grid with 'length' amino acids
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)

        # discard the protein and re-generate if it doesn't have full length
        while coord_vec[-1].x == 0:
            grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)        

        # create interarction matrix and empty arrays, start with temperature T_i
        J = energy.random_exchange_matrix()

        ergs = np.empty(T_steps, np.double)

        geo_dist = np.empty(T_steps, np.double)

        dT = (T_i - T_f)/T_steps
        T = T_i

        # calculate mean erg of protein and eucledian distance between first and
        # last amino acid for each temperature
        for i in range(T_steps):
            ergs_at_T = np.empty(num_at_T, np.double)
            geo_at_T = np.empty(num_at_T, np.double)

            for k in range(num_at_T):
                grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, T)
                x_diff = coord_vec[-1].x - coord_vec[0].x
                y_diff = coord_vec[-1].y - coord_vec[0].y

                ergs_at_T[k] = energy.total_erg_per_site(grid, coord_vec, J)
                geo_at_T[k] = np.sqrt(x_diff**2 + y_diff**2)

            ergs[i] = np.mean(ergs_at_T)
            geo_dist[i] = np.mean(geo_at_T)

            T -= dT

        delta_ergs_avg += (ergs-ergs[0])/runs
        geo_dist_avg += geo_dist/runs
        print(runIndex+1, "/", runs)
    
    # plot mean delta energy over temperature
    fig, ax = plt.subplots()
    ax.plot(np.asarray(range(T_steps)), delta_ergs_avg, label=f"$L={length}$")
    # ax.fill_between(np.asarray(range(T_steps)), ergs-d_ergs, ergs+d_ergs,
                    # alpha=0.5)
    ax.set_xlabel("T-Step")
    ax.set_ylabel("Total delta energy per site $\\Delta E/N$")
    ax.legend()
    ax2 = ax.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, " +
                 f"{num_at_T} steps per T, {runs} realisations")
    if path != "":
        fig.savefig(path+f"/annealing_energy_avg_{num_at_T}_l_{length}_steps" +
                    f"_{num_at_T*T_steps}_{runs}_runs.pdf")
    
    # plot mean eucledian distance over temperature
    fig_geo, ax_geo = plt.subplots()
    ax_geo.plot(np.asarray(range(T_steps)), geo_dist_avg, label=f"$L={length}$")
    # ax_geo.fill_between(np.asarray(range(T_steps)), geo_dist-d_geo_dist,
                        # geo_dist+d_geo_dist, alpha=0.5)
    ax_geo.set_xlabel("T-Step")
    ax_geo.set_ylabel("Euclidean distance between first and last amino acid")
    ax_geo.legend()
    ax_geo2 = ax_geo.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax_geo2.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, " +
                 f"{num_at_T} steps per T, {runs} realisations")
    if path != "":
        fig_geo.savefig(path+f"/geometric_distance_avg_{num_at_T}_l_{length}_" +
                    f"steps_{num_at_T*T_steps}_{runs}_runs.pdf")

    return delta_ergs_avg, geo_dist_avg, fig, ax, ax2, fig_geo, ax_geo, ax_geo2


def averaged_annealing(length, T_steps, num_at_T, T_i, T_f, path):
    """
    Calculate and plot mean mean erg of protein and eucledian distance between
    first and last amino acid for each temperature from T_i to T_f in T_steps
    """
    # create protein on grid with 'length' amino acids
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)

    # discard the protein and re-generate if it doesn't have full length
    while coord_vec[-1].x == 0:
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)

    # plot initial protein
    if path != "":
        figPrev, axPrev = randomwalk.plot_protein(
            coord_vec, length/3,
            path+f"/protein_init_l_{length}_steps_{num_at_T*T_steps}" +
            "_annealing.pdf")
    else:
        figPrev, axPrev = randomwalk.plot_protein(coord_vec, length/3, "")

    # create interarction matrix and empty arrays, start with temperature T_i
    J = energy.random_exchange_matrix()

    ergs = np.empty(T_steps, np.double)
    d_ergs = np.empty(T_steps, np.double)

    geo_dist = np.empty(T_steps, np.double)
    d_geo_dist = np.empty(T_steps, np.double)

    dT = (T_i - T_f)/T_steps
    T = T_i

    # calculate mean erg of protein and eucledian distance between first and
    # last amino acid for each temperature
    for i in range(T_steps):
        ergs_at_T = np.empty(num_at_T, np.double)
        geo_at_T = np.empty(num_at_T, np.double)

        for k in range(num_at_T):
            grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, T)
            x_diff = coord_vec[-1].x - coord_vec[0].x
            y_diff = coord_vec[-1].y - coord_vec[0].y

            ergs_at_T[k] = energy.total_erg_per_site(grid, coord_vec, J)
            geo_at_T[k] = np.sqrt(x_diff**2 + y_diff**2)

        ergs[i] = np.mean(ergs_at_T)
        d_ergs[i] = np.std(ergs_at_T)

        geo_dist[i] = np.mean(geo_at_T)
        d_geo_dist[i] = np.std(geo_at_T)

        T -= dT

    # plot finished protein
    if path != "":
        randomwalk.plot_protein(
            coord_vec, length/3,
            path+f"/protein_final_l_{length}_steps_{num_at_T*T_steps}_" +
            "annealing.pdf")

    # plot mean energy over temperature
    fig, ax = plt.subplots()
    ax.plot(np.asarray(range(T_steps)), ergs, label=f"$L={length}$")
    # ax.fill_between(np.asarray(range(T_steps)), ergs-d_ergs, ergs+d_ergs,
                    # alpha=0.5)
    ax.set_xlabel("T-Step")
    ax.set_ylabel("Total energy per site $E/N$")
    ax.legend()
    ax2 = ax.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, " +
                 f"{num_at_T} steps per T")
    if path != "":
        fig.savefig(path+f"/annealing_energy_avg_{num_at_T}_l_{length}_steps" +
                    f"_{num_at_T*T_steps}.pdf")

    # plot mean eucledian distance over temperature
    fig_geo, ax_geo = plt.subplots()
    ax_geo.plot(np.asarray(range(T_steps)), geo_dist, label=f"$L={length}$")
    # ax_geo.fill_between(np.asarray(range(T_steps)), geo_dist-d_geo_dist,
                        # geo_dist+d_geo_dist, alpha=0.5)
    ax_geo.set_xlabel("T-Step")
    ax_geo.set_ylabel("Euclidean distance between first and last amino acid")
    #ax_geo.semilogx()
    ax_geo.legend()
    ax2geo = ax_geo.secondary_xaxis("top", functions=(
            lambda step : T_i - step*dT, lambda T : (T_i-T)/dT
    ))
    ax2geo.set_xlabel("Temperature $T$")

    ax.set_title(f"{num_at_T*T_steps} monte carlo steps, $dT={dT}$, " +
                 f"{num_at_T} steps per T")
    if path != "":
        fig_geo.savefig(path+f"/geometric_distance_avg_{num_at_T}_l_{length}_" +
                    f"steps_{num_at_T*T_steps}.pdf")

    return ergs, grid, coord_vec, fig, ax, ax2, geo_dist, fig_geo, ax_geo, ax2geo, figPrev, axPrev
