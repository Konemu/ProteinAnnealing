# libs
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numba import jit, njit, prange


# generate random exchange matrix elements with exp. value 3 and stddev 1/sqrt(2)
@njit
def random_exchange_matrix():
    mu = -3.0
    sig = 1.0/np.sqrt(2)
    # generate non-symmetrical matrix following requested distribution
    J_non_symm = normal(mu, sig, (20, 20))
    # use the fact that the sum of two normally distributed random numbers is normally distributed
    # with mu = mu_1 + mu_2 and sig^2 = sig_1^2 + sig_2^2
    # (here mu_1 = mu_2 so we divide by two to recover the original distribution)
    J_symm = (J_non_symm + J_non_symm.T) / 2
    return J_symm

# helper function for eigenvalue_statistics to profit from numba
@njit(parallel=True)
def parallel_ev_calc(runs, eigenvalues):
    for k in prange(runs):
        # generate a random exchange matrix runs-times
        J = random_exchange_matrix()
        ev, discard = np.linalg.eigh(J) # we don't need the eigenvectors! this should use LAPACK or MKL btw :^) thanks numba
        for i in range(k*20, (k+1)*20):
            eigenvalues[i] = ev[i % 20] # store evs


def eigenvalue_statistics(runs, path):
    eigenvalues = np.empty(runs*20, dtype=np.double)
    parallel_ev_calc(runs, eigenvalues)

    ev_mean = np.mean(eigenvalues)
    ev_std = np.std(eigenvalues)
    
    # make a nice picture
    fig, ax = plt.subplots()
    ax.hist(eigenvalues, density=True, bins=1000, label="Eigenvalues")
    ax.vlines(ev_mean, 0, 1, label=f"mean = {np.round(ev_mean, 3)}", color="red")
    ax.vlines([ev_mean-ev_std, ev_mean+ev_std], 0, 1, label = f"1$\sigma = {np.round(ev_std, 1)}$-width", color="green")
    ax.set_ylabel("Eigenvalue occurence $N_\lambda$")
    ax.set_xlabel("Eigenvalue $\lambda$")
    ax.legend()
    ax.set_ylim([0, 0.2])
    ax.set_title(f"{runs} runs")

    fig.savefig(path)

    return fig, ax, eigenvalues, ev_mean, ev_std # return everything!!1!


    