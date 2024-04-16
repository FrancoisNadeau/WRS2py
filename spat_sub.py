import numpy as np

def spat_sub(x, theta):
    xx = np.copy(x)
    for i in range(x.shape[1]):
        xx[:, i] = x[:, i] - theta[i]
    xx = xx ** 2
    temp = np.sqrt(np.apply_along_axis(np.sum, 1, xx))
    val = np.mean(temp)
    return val