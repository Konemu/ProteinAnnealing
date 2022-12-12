# libs
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# code
import randomwalk


@njit
def mean_sq_pos(coord_vec, steps):
    x2_mean = 0.0
    y2_mean = 0.0
    for coords in coord_vec:
        x2_mean += coords.x**2
        y2_mean += coords.y**2
    x2_mean /= steps
    y2_mean /= steps

    x2_std = 0.0
    y2_std = 0.0
    for coords in coord_vec:
        x2_std += (coords.x**2 - x2_mean)**2
        y2_std += (coords.y**2 - y2_mean)**2
    mean_distance = np.sqrt(x2_mean + y2_mean)

    return x2_mean, y2_mean, x2_std, y2_std, mean_distance


@njit(parallel=True)
def sq_pos_stats_parallel_self_avoiding(runs, dim, steps):
    xs              = np.zeros(runs*steps)
    ys              = np.zeros(runs*steps)
    x2_means        = np.zeros(runs, dtype=np.double)
    y2_means        = np.zeros(runs, dtype=np.double)
    x2_stds         = np.zeros(runs, dtype=np.double)
    y2_stds         = np.zeros(runs, dtype=np.double)
    mean_distances  = np.zeros(runs, dtype=np.double)
    eff_runs = runs
    for k in prange(runs):
        grid, coord_vec = randomwalk.self_avoiding_walk(dim, steps)
        if len(coord_vec) != steps:
            eff_runs -= 1
            break
        for i in range(k*steps, (k+1)*steps):
            xs[i] = coord_vec[i % steps].x
            ys[i] = coord_vec[i % steps].y
        x2_mean, y2_mean, x2_std, y2_std, mean_distance = mean_sq_pos(coord_vec, steps)
        x2_means[k]         = x2_mean
        y2_means[k]         = y2_mean
        x2_stds[k]          = x2_std
        y2_stds[k]          = y2_std
        mean_distances[k]   = mean_distance
    return xs, ys, x2_means, y2_means, x2_stds, y2_stds, mean_distances, eff_runs


def mean_sq_pos_stats_self_avoiding(runs, dim, steps, path):
    xs, ys, x2_means, y2_means, x2_stds, y2_stds, mean_distances, eff_runs = sq_pos_stats_parallel_self_avoiding(runs, dim, steps)

    fig, ax = plt.subplots()
    ax.hist2d(x2_means, y2_means, bins= 100, range=[[0,np.sqrt(dim)], [0, np.sqrt(dim)]])
    ax.set_ylim([0, np.sqrt(dim)])
    ax.set_xlim([0, np.sqrt(dim)])
    ax.set_aspect('equal')
    ax.set_ylabel("$y^2$")
    ax.set_xlabel("$x^2$")
    ax.set_title("Mittlere quadratische Position")
    fig.savefig(path + "/mean_sq_pos_self_av.pdf")

    print(x2_means.mean(), y2_means.mean(), mean_distances.mean(), eff_runs)

    return fig,ax, xs, ys, x2_means, y2_means, x2_stds, y2_stds, mean_distances, eff_runs






