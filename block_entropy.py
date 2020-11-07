# Compute the Shannon entropy of a distribution of points
# in the phase space
#  
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 

import numpy as np

def block_entropy(data, epsilon=1e-2):
    """
        Compute Shannon entropy of the distribution of 
        phase-space points by discretizing the space 
        into unit cell of size epsilon. The result is 
        not sensitive to the value of epsilon.

        S = \sum_i^N p_i log2(1/p_i), where p_i is the 
        occupancy probability, p_i = N_i/N. N is the 
        total number of points and N_i is the number of 
        points located inside the i-th unit cell.

        Input data is of size ndim*N
        ndim: dimension of the phase space
    """
    ndim,N = data.shape
    coords = np.zeros((N, ndim))
    for i in range(ndim): # iterate through the dimensions
        x = data[i]-np.min(data[i]) # shift the data points
        x = np.floor(x/(epsilon)).astype(int) # discretize the space
        coords[:,i] = x

    # find unique coordinates and the counts for each coordinate
    coord_unique, counts = np.unique(coords, axis = 0, return_counts = True)
    p = counts/N # occupancy probability
    
    return -np.sum(p*np.log2(p))