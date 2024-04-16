import numpy as np
from numpy.linalg import eig

def gvar(m):
    m = elimna(m)
    temp = np.var(m)
    eigen_values = eig(temp)[0]
    gvar = np.prod(eigen_values)
    return gvar