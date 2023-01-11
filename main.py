# libs
from numba import set_num_threads
# adjust this to your preference. at least one less than cpu cores is recommended,
# since numba code can't easily be interrupted and you might lock up your system if
# something goes wrong
set_num_threads(8)

# code
import randomwalk
import energy
import randomwalk_statistics
import a5


def main():
    #energy.eigenvalue_statistics(1000000, "data/eigenvalue_distribution.pdf")
    #randomwalk_statistics.mean_sq_pos_stats_both(10000, 100, 100, "data")
    #energy.evolve_protein_plot_energy(100, 100000, 1, "data")
    #energy.animated_monte_carlo(50, 1000, 1, 1, 8, "data")
    a5.annealing(20, 100000, "data")

if __name__=="__main__":
    main()