# libs
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# code
import randomwalk
import energy
import randomwalk_statistics

def annealing(length, mc_steps, path):
    '''
        This function represents a5. It is essentially the same as a4, but the temperature 
        is now an additional variable. Thus it changes with each monte carlo step. 

        For this the evolve_protein_plot_energy function in energy.py has been slightly 
        altered to adapt for this change of double <-> array. 
    '''
    # this is not the way
    """
    Temps = np.asarray(np.linspace(1, 10, 100))
    Temps = Temps[::-1] # invert array
    
    ergs_save_mean = np.zeros((len(Temps), mc_steps), dtype=np.double)
    coord_vec_save = np.array((len(Temps), mc_steps), dtype=np.double)
    
    hashtag_same_temp = 10


    # Keine Ahnung was genau die Aufgabe will. In ergs_save_mean wird jetzt jedenfalls 
    # zu verschiedenen Temperaturen in den Zeilen die ganzen gemittelten Monte-Carlo Steps
    # gespeichert. Ich weiß nicht wie Schomi sich das vorstellen will, was genau da gemittelt
    # werden soll. Die Temperaturen untereinander ist ja dumm, aber die unterschiedlichen
    # Steps mitteln, hört sich für mich auch erstmal nicht so sinnvoll an.

    # Das Gleiche soll nochmal für coord_vec_save entstehen.
    for t in range(len(Temps)):
        for i in range(1,hashtag_same_temp):
            ergs, coord_vec = energy.evolve_protein_plot_energy(length=length, 
                                mc_steps=mc_steps, T=Temps[t], path='', a5 = True)
            print("T=",Temps[t])
            for j in range(len(ergs)):
                ergs_save_mean[t][j] += ergs[j]/hashtag_same_temp
    """

    # Erster Part der Aufgabe 5. Einfach ein Temp Plot des annealing
    Temps = np.asarray(np.linspace(10, 0.01, mc_steps))
    #Temps = Temps[::-1] # invert array
    ergs, coord_vec = energy.evolve_protein_plot_energy_var_temp(length, mc_steps, Temps, path)

    #PLOTS
    fig, ax = plt.subplots()
    ax.plot(Temps, ergs, label=f"$L={length}$") # plot energy
    ax.invert_xaxis()
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Total energy $E$")
    ax.legend()
    ax.set_title(f"{mc_steps} monte carlo steps")
    if path != "":
        fig.savefig(path+f"/annealing_energy_l_{length}_steps_{mc_steps}.pdf")
    plt.show()

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

<<<<<<< HEAD
    fig_geo, ax_geo = plt.subplots()
    ax_geo.plot(np.asarray(range(T_steps)), geo_distance, label=f"$L={length}$")
    ax_geo.set_xlabel("T-Step")
    ax_geo.set_ylabel("Euclidean distance between first and last amino acid")
    ax_geo.semilogx()
    ax_geo.legend()
=======
    return ergs, grid, coord_vec, fig, ax, ax2, figPrev, axPrev
>>>>>>> 3df0cdbc9d7cee241933bb172eab0a8950b3cf84

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

    return ergs, grid, coord_vec, fig, ax, ax2, geo_distance, fig_geo, ax_geo, ax2_geo
