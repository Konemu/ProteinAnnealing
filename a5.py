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
            print(Temps[t])
            for j in range(len(ergs)):
                ergs_save_mean[t][j] += ergs[j]/hashtag_same_temp
    """

    # Erster Part der Aufgabe 5. Einfach ein Temp Plot des annealing
    Temps = np.asarray(np.linspace(1, 10, mc_steps))
    Temps = Temps[::-1] # invert array
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

    return ergs, coord_vec

