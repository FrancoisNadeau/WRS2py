import numpy as np
from scipy.stats import median_absolute_deviation

def med1way_crit(n, alpha=0.05, iter=1000, TEST=None, SEED=True):
    J = len(n)
    x = []
    w = np.zeros(J)
    xbar = np.zeros(J)
    if SEED:
        np.random.seed(2)
    chk = np.zeros(iter)
    grp = np.arange(1, J+1)
    for it in range(iter):
        for j in range(J):
            x.append(np.random.normal(size=n[j]))
            w[j] = 1 / median_absolute_deviation(x[grp[j]])**2
            xbar[j] = np.median(x[grp[j]])
            n[j] = len(x[grp[j]])
        u = np.sum(w)
        xtil = np.sum(w*xbar) / u
        chk[it] = np.sum(w*(xbar-xtil)**2) / (J-1)
    chk = np.sort(chk)
    iv = round((1-alpha) * iter)
    crit_val = chk[iv]
    pval = None
    if TEST is not None:
        pval = np.sum((TEST <= chk)) / iter
    return {'crit_val': crit_val, 'alpha': alpha, 'p_value': pval}