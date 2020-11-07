# Gram–Schmidt orthogonalization of the columns of input matrix A
# Written by Shiyuan Hu <shiyuan.hu@nyu.edu>, Nov. 6, 2020 

import numpy as np

def gsr(A):
    def proj(v,u):
        """
            compute the projection of v onto u
        """
        prefc = v.dot(u)/u.dot(u)
        return prefc*u

    _,ncol = A.shape
    vnorms = np.zeros(ncol) 
    vnorms[0] = np.linalg.norm(A[:,0]) # normalized the first column

    for i in range(1,ncol): # iteration starts at the second column
        Ai = A[:,i]
        Ai_new = np.copy(Ai)
        for j in range(0, i): # subtract the projection of Ai on the previous orthonormal columns 
            Aj = A[:,j]
            Ai_new = Ai_new-proj(Ai,Aj)
        vnorms[i] = np.linalg.norm(Ai_new)
        A[:,i] = Ai_new
    return A/vnorms, vnorms