# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:38:41 2023

@author: Luis-
"""

import numpy as np
# from numba import njit, prange


def barrier(mc_steps, ergs):
    """
    Return array containing all meta-stabel energies (initial energy included)
    for a given evolved protein over the number of monte-carlo-steps / time.
    """
    barrier_array = np.array([ergs[0]])
    barrier = 0

    for i in range(1, mc_steps):

        if ergs[i] - ergs[i-1] > 0:
            barrier += ergs[i] - ergs[i-1]

        elif barrier > 0 and ergs[i] - ergs[i-1] < 0:
            # prüfe ob zustand bereits in array vorhanden
            if abs(barrier_array[0] - ergs[i]) not in barrier_array:
                barrier_array = np.append(
                    barrier_array, ergs[i-1])

            barrier = 0

    print("Test", barrier_array)

    return barrier_array
