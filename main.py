# libs
from numba import set_num_threads
# adjust this to your preference. at least one less than cpu cores is recommended,
# since numba code can't easily be interrupted and you might lock up your system if
# something goes wrong
set_num_threads(10)
import numpy as np

# code
import randomwalk
import energy
import randomwalk_statistics


def main():
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(100, 100)
    while coord_vec[-1].x == 0:
        grid, coord_vec = randomwalk.self_avoiding_walk_protein(100, 100)
    randomwalk.plot_protein(coord_vec, 25, "data/protein_test_plot.pdf")
    J = energy.random_exchange_matrix()
    for k in range(1000000):
        grid, coord_vec = energy.monte_carlo_step(grid, coord_vec, J, 1)
    randomwalk.plot_protein(coord_vec, 25, "data/protein_1000000_folds.pdf")
    #energy.eigenvalue_statistics(1000000, "data/eigenvalue_distribution.pdf")
    #randomwalk_statistics.mean_sq_pos_stats_both(10000, 100, 100, "data")


if __name__=="__main__":
    main()