import numpy as np
import pandas as pd
from scipy.stats import chi2

def out(x, cov_fun=cov.mve, SEED=True, xlab="X", ylab="Y", qval=0.975, crit=None, plotit=False, *args, **kwargs):
    if SEED:
        np.random.seed(12)
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, list):
        raise ValueError("Data cannot be stored in list mode")
    nrem = x.shape[0]
    if not isinstance(x, np.ndarray):
        dis = (x - np.median(x, axis=0, nan_policy='omit'))**2 / np.median(np.abs(x - np.median(x, axis=0, nan_policy='omit')), axis=0)**2
        if crit is None:
            crit = np.sqrt(chi2.ppf(qval, 1))
        vec = np.arange(1, x.shape[0]+1)
    if isinstance(x, np.ndarray):
        mve = cov_fun(elimna(x))
        dis = mahalanobis(x, mve['center'], mve['cov'])
        if crit is None:
            crit = np.sqrt(chi2.ppf(qval, x.shape[1]))
        vec = np.arange(1, x.shape[0]+1)
    dis[np.isnan(dis)] = 0
    dis = np.sqrt(dis)
    chk = np.where(dis > crit, 1, 0)
    id = vec[chk == 1]
    keep = vec[chk == 0]
    if isinstance(x, np.ndarray):
        if x.shape[1] == 2 and plotit:
            plt.plot(x[:, 0], x[:, 1], xlab=xlab, ylab=ylab, type="n")
            flag = np.repeat(True, x.shape[0])
            flag[id] = False
            plt.scatter(x[flag, 0], x[flag, 1])
            if np.sum(~flag) > 0:
                plt.scatter(x[~flag, 0], x[~flag, 1], marker="*")
    if not isinstance(x, np.ndarray):
        outval = x[id]
    if isinstance(x, np.ndarray):
        outval = x[id, :]
    n = x.shape[0]
    n_out = len(id)
    return {'n': n, 'n_out': n_out, 'out.val': outval, 'out.id': id, 'keep': keep, 'dis': dis, 'crit': crit}