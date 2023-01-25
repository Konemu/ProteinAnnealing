# ProteinAnnealing
Simplistic 2D monte carlo / simulated annealing protein folding project for the Computational Physics lecture @ TU Braunschweig

This project implements a simple monte-carlo time evolution of a 2D-protein initialised as a self-avoiding random walk on a 2D grid.

Peptide interaction contributions to the total energy are determined by randomly generated normal-distributed interaction matrix elements.

Routines for plotting and animation are provided.

Simulated annealing is implemented as well, extending the constant-temperature monte carlo iteration and yielding better energy minimisation for the same number of steps.

Some simple statistic analyses of quantities of interest (cf. exercises) are available.

The jupyter notebook nb.py was meant for the in-class presentation of results and demonstrates the implemented functionalities.

Required packages are among others:
- Python 3.10
- Numba 0.56
- Matplotlib, numpy, scipy

