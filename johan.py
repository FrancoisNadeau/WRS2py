import numpy as np
from scipy.stats import chi2

def johan(cmat, vmean, vsqse, h, alpha=0.05):
    yvec = np.matrix(vmean).T
    if not isinstance(vsqse, np.matrix):
        vsqse = np.diag(vsqse)
    test = cmat @ vsqse @ cmat.T
    invc = np.linalg.inv(test)
    test = yvec.T @ cmat.T @ invc @ cmat @ yvec
    R = vsqse @ cmat.T @ invc @ cmat
    A = np.sum(np.diag((np.diag(R))**2 / (h - 1)))
    df = cmat.shape[0]
    crit = chi2.ppf(1 - alpha, df)
    crit = crit + (crit / (2 * df)) * A * (1 + 3 * crit / (df + 2))
    return {'teststat': test[0, 0], 'crit': crit}