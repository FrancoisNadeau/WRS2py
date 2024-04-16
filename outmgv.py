import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

def outmgv(x, y=None, plotit=True, outfun=outbox, se=True, op=1, cov_fun=rmba, xlab="X", ylab="Y", SEED=True, STAND=False, **kwargs):
    if y is None:
        m = x
    else:
        m = np.c_[x, y]
    m = elimna(m)
    m = np.asmatrix(m)
    nv = m.shape[0]
    temp = mgvar(m, se=se, op=op, cov_fun=cov_fun, SEED=SEED)
    temp[np.isnan(temp)] = 0
    if m.shape[1] == 1:
        temp2 = outpro(m)
        nout = temp2['n.out']
        keep = temp2['keep']
        temp2 = temp2['out.id']
    elif m.shape[1] > 1:
        if m.shape[1] == 2:
            temp2 = outfun(temp, **kwargs)['out.id']
        elif m.shape[1] > 2:
            temp2 = outbox(temp, mbox=True, gval=np.sqrt(chi2.ppf(0.975, m.shape[1])))['out.id']
        vec = np.arange(1, m.shape[0]+1)
        flag = np.repeat(True, m.shape[0])
        flag[temp2] = False
        vec = vec[flag]
        vals = np.arange(1, m.shape[0]+1)
        keep = vals[flag]
        if plotit and m.shape[1] == 2:
            x = m[:, 0]
            y = m[:, 1]
            plt.plot(x, y, 'n', label=xlab, ylabel=ylab)
            flag = np.repeat(True, len(y))
            flag[temp2] = False
            plt.scatter(x[flag], y[flag], marker='*')
            plt.scatter(x[temp2], y[temp2], marker='o')
    nout = 0
    if not np.isnan(temp2[0]):
        nout = len(temp2)
    return {'n': nv, 'n.out': nout, 'out.id': temp2, 'keep': keep}