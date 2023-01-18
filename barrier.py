# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:38:41 2023

@author: Luis-
"""

import numpy as np
# from numba import njit, prange


def barrier(mc_steps, ergs):
    """
    Returns array containing all energy barriers
    """
    meta_stable_array = np.array([ergs[0]])
    barrier_array = np.array([])
    index_of_barrier = []
    time_to_barrier = -1
    barrier = 0
    


    for i in range(1, mc_steps):
        if ergs[i] == ergs[i-1] and ergs[i] not in meta_stable_array:
            meta_stable_array = np.append(meta_stable_array, ergs[i])


    for i in range(1, len(meta_stable_array)):
        if meta_stable_array[i] > meta_stable_array[i-1]:
            barrier += abs(meta_stable_array[i] - meta_stable_array[i-1])

        elif meta_stable_array[i] < meta_stable_array[i-1] and barrier != 0:
            barrier_array = np.append(barrier_array, barrier)
            index_of_barrier.append(i)  
            barrier = 0

    for i in range(1, mc_steps):
        if len(index_of_barrier)>0 and np.abs((ergs[i] - meta_stable_array[index_of_barrier[0]-1]) / ergs[i]) < 1e-10:
            time_to_barrier = i
            print('MC Schritte bis zur ersten Barriere: ' + str(i))
            break

    print("Energiebarrieren:", barrier_array)

    return barrier_array, time_to_barrier

def time_to_metastable(ergs):
    meta_stable_array = np.array([ergs[0]])
    meta_stable_indices = []
    barrier_array = np.array([])
    index_of_barrier = []
    barrier = 0
    mc_steps = len(ergs)

    step_first_metastable = -1


    for i in range(1, mc_steps):
        if ergs[i] == ergs[i-1] and ergs[i] not in meta_stable_array:
            meta_stable_array = np.append(meta_stable_array, ergs[i])
            meta_stable_indices.append(i)

    for i in range(1, len(meta_stable_array)):
        if meta_stable_array[i] > meta_stable_array[i-1]:
            barrier += abs(meta_stable_array[i] - meta_stable_array[i-1])

        elif meta_stable_array[i] < meta_stable_array[i-1] and barrier != 0:
            barrier_array = np.append(barrier_array, barrier)
            index_of_barrier.append(i)  
            barrier = 0

    for i in range(len(meta_stable_indices)-1):
        curr_erg = ergs[meta_stable_indices[i]]
        next_erg = ergs[meta_stable_indices[i+1]]
        if curr_erg < next_erg:
            step_first_metastable = meta_stable_indices[i+1]
            break


    return step_first_metastable
