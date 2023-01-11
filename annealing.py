# libs
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# code
import randomwalk
import energy
import randomwalk_statistics

def annealing(length, T_init, T_end, DeltaT, hash_same_temp, path):
    '''
        This function represents a5. It is essentially the same as a4, but the temperature 
        is now an additional variable. Thus it changes with each monte carlo step. 

        For this the evolve_protein_plot_energy function in energy.py has been slightly 
        altered to adapt for this change of double <-> array. 
    '''
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    while coord_vec[-1].x == 0: # discard the protein and re-generate if it doesn't have full length
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(length, length)
    J = energy.random_exchange_matrix() # generate a random exchange matrix
    
    
    T = T_init

    total_length = int((T_init-T_end)/DeltaT)

    ergs_save_temp = np.zeros(total_length, dtype = np.double)
    for t in range(total_length):
        for i in range(hash_same_temp):
            grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, T) # perform mc steps
            ergs_value = energy.total_erg_per_site(grid, coord_vec, J)
            
            ergs_save_temp[t] += ergs_value/hash_same_temp
        T -= DeltaT

    temps = np.asarray(np.linspace(T_end, 10, total_length))
    temps = temps[::-1] # invert array

    fig, ax = plt.subplots()
    ax.plot(temps, ergs_save_temp)
    ax.invert_xaxis()
    plt.show()


    # # Erster Part der Aufgabe 5. Einfach ein Temp Plot des annealing
    # Temps = np.asarray(np.linspace(10, 0.01, mc_steps))
    # #Temps = Temps[::-1] # invert array
    # ergs, coord_vec = energy.evolve_protein_plot_energy_var_temp(length, mc_steps, Temps, path)

    # #PLOTS
    # fig, ax = plt.subplots()
    # ax.plot(Temps, ergs, label=f"$L={length}$") # plot energy
    # ax.invert_xaxis()
    # ax.set_xlabel("Temperature $T$")
    # ax.set_ylabel("Total energy $E$")
    # ax.legend()
    # ax.set_title(f"{mc_steps} monte carlo steps")
    # if path != "":
    #     fig.savefig(path+f"/annealing_energy_l_{length}_steps_{mc_steps}.pdf")
    # plt.show()

    return ergs_save_temp, coord_vec

