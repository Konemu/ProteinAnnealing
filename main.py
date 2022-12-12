# libs
from numba import set_num_threads
# adjust this to your preference. at least one less than cpu cores is recommended,
# since numba code can't easily be interrupted and you might lock up your system if
# something goes wrong
set_num_threads(10) 

# code
import randomwalk
import energy
import randomwalk_statistics


def main():
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(10, 30)
    randomwalk.plot_protein(coord_vec, 10, "data/protein_test_plot.pdf")
    energy.eigenvalue_statistics(1000000, "data/eigenvalue_distribution.pdf")
    randomwalk_statistics.mean_sq_pos_stats_both(1000, 100, 100, "data")


if __name__=="__main__":
    main()