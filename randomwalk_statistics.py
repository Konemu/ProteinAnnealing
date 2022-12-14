# libs
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

# code
import randomwalk


# generate squared average positions for self-avoiding walk
# note: walks may be discarded if they "curl up" prematurely!
# another approach might be to let this run until runs is reached.
# ^^^^ feel free to change!
@njit(parallel=True)
def sq_pos_stats_parallel_self_avoiding(runs, dim, steps):
    x2s              = np.zeros(steps)
    y2s              = np.zeros(steps)
    eff_runs = runs # remove discarded runs

    for k in prange(runs):
        grid, coord_vec = randomwalk.self_avoiding_walk(dim, steps)
        if coord_vec[-1].x == 0: # if last coord has coordinate 0, we have an invalid walk
            eff_runs -= 1
            continue
        for i in range(steps):
            x2s[i] += coord_vec[i].x**2
            y2s[i] += coord_vec[i].y**2
        
    return x2s/eff_runs, y2s/eff_runs, eff_runs # calculate mean correctly (divide by eff_runs)


# generate squared average positions for random walk
@njit(parallel=True)
def sq_pos_stats_parallel(runs, dim, steps):
    x2s              = np.zeros(steps)
    y2s              = np.zeros(steps)

    for k in prange(runs):
        grid, coord_vec = randomwalk.random_walk(dim, steps)
        for i in range(steps):
            x2s[i] += coord_vec[i].x**2
            y2s[i] += coord_vec[i].y**2
        
    return x2s/runs, y2s/runs



# at this time, this "only" plots the distance from the origin as a function
# of time = step for a self-avoiding walk
# feel free to add cool stuff if you have ideas
#
# runs: # of walks to generate including ones that will potentially be discarded
# dim: max. x-coordinate
# steps: length of walks to be generated
# path: where to save the picture
def mean_sq_pos_stats_self_avoiding(runs, dim, steps, path):
    x2s, y2s, eff_runs = sq_pos_stats_parallel_self_avoiding(runs, dim, steps)

    distance = np.sqrt(x2s + y2s)
    step = np.array([i for i in range(1, steps + 1)])

    fig, ax = plt.subplots()
    ax.plot(step, distance, label=f"Length $L$ = {steps}")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Mean distance of walk from origin $d$")
    ax.set_title(f"Self-avoiding, {eff_runs} runs, {2*dim + 1}x{2*dim + 1}-grid")
    ax.legend()
    fig.savefig(path + "/mean_pos_avoid.pdf")

    print(eff_runs)

    return fig, ax, x2s, y2s, eff_runs


# at this time, this "only" plots the distance from the origin as a function
# of time = step for a random walk
# feel free to add cool stuff if you have ideas
#
# runs: # of walks to generate
# dim: max. x-coordinate
# steps: length of walks to be generated
# path: where to save the picture
def mean_sq_pos_stats(runs, dim, steps, path):
    x2s, y2s = sq_pos_stats_parallel(runs, dim, steps)

    distance = np.sqrt(x2s + y2s)
    step = np.array([i for i in range(1, steps + 1)])

    fig, ax = plt.subplots()
    ax.plot(step, distance, label=f"Length $L$ = {steps}")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Mean distance of walk from origin $d$")
    ax.set_title(f"Non-self-avoiding, {runs} runs, {2*dim + 1}x{2*dim + 1}-grid")
    ax.legend()
    fig.savefig(path + "/mean_pos.pdf")


    return fig, ax, x2s, y2s


# like both of the above but combined.
def mean_sq_pos_stats_both(runs, dim, steps, path):
    fig1, ax1, x2s_r, y2s_r = mean_sq_pos_stats(runs, dim, steps, path)
    fig2, ax2, x2s_sa, y2s_sa, eff_runs = mean_sq_pos_stats_self_avoiding(runs, dim, steps, path)
    distance_r = np.sqrt(x2s_r + y2s_r)
    distance_sa = np.sqrt(x2s_sa + y2s_sa)
    step = np.array([i for i in range(1, steps + 1)])

    fig, ax = plt.subplots()
    ax.plot(step, distance_r, label = f"non-avoiding, $n = {runs}$")
    ax.plot(step, distance_sa, label = f"self-avoiding, $n = {eff_runs}$")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Mean distance of walk from origin $d$")
    ax.set_title(f"{2*dim + 1}x{2*dim + 1}-grid")
    ax.legend()

    fig.savefig(path + "/mean_pos_comp.pdf")





