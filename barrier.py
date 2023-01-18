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
    barrier = 0

    print('hi')

    for i in range(1, mc_steps):
        if ergs[i] == ergs[i-1] and ergs[i] not in meta_stable_array:
            meta_stable_array = np.append(meta_stable_array, ergs[i])

    for i in range(1, len(meta_stable_array)):
        if meta_stable_array[i] > meta_stable_array[i-1]:
            barrier += abs(meta_stable_array[i] - meta_stable_array[i-1])

        elif meta_stable_array[i] < meta_stable_array[i-1] and barrier != 0:
            barrier_array = np.append(barrier_array, barrier)
            index_of_barrier.append(i)  
            print(index_of_barrier)
            barrier = 0

    for i in range(1, mc_steps):
        if len(index_of_barrier)>0 and ergs[i] == meta_stable_array[index_of_barrier[0]]:
            print('MC Schritte bis zur ersten Barriere: ' + str(i))
        break

    print("Energiebarrieren:", barrier_array)

    return barrier_array
