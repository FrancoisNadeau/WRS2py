import numpy as np
from scipy.stats import t

def smmvalv2(dfvec, iter=10000, alpha=0.05, SEED=True):
    if SEED:
        np.random.seed(1)
    
    dfv = len(dfvec) / sum(1 / dfvec)
    vals = np.empty(iter)
    tvals = np.empty(iter)
    J = len(dfvec)
    z = np.empty((iter, J))
    
    for j in range(J):
        z[:, j] = t.rvs(dfvec[j], size=iter)
    
    vals = np.max(z, axis=1)
    vals = np.sort(vals)
    ival = round((1 - alpha) * iter)
    qval = vals[ival]
    
    return qval