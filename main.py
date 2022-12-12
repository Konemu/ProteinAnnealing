# libs

# code
import randomwalk
import energy


def main():
    grid, coord_vec = randomwalk.self_avoiding_walk_protein(10, 30)
    randomwalk.plot_protein(coord_vec, 10, "data/protein_test_plot.pdf")
    energy.eigenvalue_statistics(1000000, "data/eigenvalue_distribution.pdf")


if __name__=="__main__":
    main()