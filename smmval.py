import numpy as np
from scipy.stats import t

def smmval(dfvec, iter=10000, alpha=0.05, SEED=True):
    if SEED:
        np.random.seed(1)
    dfv = len(dfvec) / sum(1 / dfvec)
    vals = np.empty(iter)
    tvals = np.empty(len(dfvec))
    J = len(dfvec)
    for i in range(iter):
        for j in range(J):
            tvals[j] = t.rvs(dfvec[j], size=1)
        vals[i] = np.max(np.abs(tvals))
    vals = np.sort(vals)
    ival = round((1 - alpha) * iter)
    qval = vals[ival]
    return qval