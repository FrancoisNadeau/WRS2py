import numpy as np

def elimna(m):
    if np.ndim(m) == 0:
        m = np.array(m)
    ikeep = np.arange(1, np.shape(m)[0] + 1)
    for i in range(np.shape(m)[0]):
        if np.sum(np.isnan(m[i, :])) >= 1:
            ikeep[i] = 0
    elimna = m[ikeep[ikeep >= 1], :]
    return elimna