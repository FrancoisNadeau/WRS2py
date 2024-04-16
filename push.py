import numpy as np

def push(mat):
    matn = np.empty_like(mat)
    Jm = mat.shape[0] - 1
    for k in range(mat.shape[1]):
        temp = mat[:, k]
        vec = np.zeros_like(temp)
        vec[1:] = temp[:Jm]
        matn[:, k] = vec
    return matn