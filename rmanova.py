import pandas as pd
import numpy as np
from scipy.stats import f

def rmanova(y, groups, blocks, tr=0.2, **kwargs):
    cols1 = y.name
    cols2 = groups.name
    cols3 = blocks.name
    dat = pd.concat([y, groups, blocks], axis=1)
    dat.columns = [cols1, cols2, cols3]
    x = dat.pivot(index=cols3, columns=cols2, values=cols1).iloc[:, 1:]
    grp = np.arange(1, len(x)+1)
    if isinstance(x, pd.DataFrame):
        J = len(grp)
        m1 = np.matrix(x.iloc[:, 0]).reshape(-1, 1)
        for i in range(1, J):
            m2 = np.matrix(x.iloc[:, i]).reshape(-1, 1)
            m1 = np.hstack((m1, m2))
    m2 = np.zeros((m1.shape[0], m1.shape[1]))
    xvec = np.ones(m1.shape[1])
    g = int(tr * m1.shape[0])
    for j in range(m1.shape[1]):
        m2[:, j] = winval(m1[:, j], tr)
        xvec[j] = np.mean(m1[:, j])
    xbar = np.mean(xvec)
    qc = (m1.shape[0] - 2 * g) * np.sum((xvec - xbar)**2)
    m3 = np.zeros((m1.shape[0], m1.shape[1]))
    m3 = np.subtract(m2, np.mean(m2, axis=1).reshape(-1, 1))
    m3 = np.subtract(m3, np.mean(m2, axis=0))
    m3 = np.add(m3, np.mean(m2))
    qe = np.sum(m3**2)
    test = qc / (qe / (m1.shape[0] - 2 * g - 1))
    
    v = winall(m1, tr=tr)['cov']
    vbar = np.mean(v)
    vbard = np.mean(np.diag(v))
    vbarj = np.ones(J)
    for j in range(J):
        vbarj[j] = np.mean(v[j, :])
    A = J * J * (vbard - vbar)**2 / (J - 1)
    B = np.sum(v * v) - 2 * J * np.sum(vbarj**2) + J * J * vbar**2
    ehat = A / B
    etil = (m1.shape[0] * (J - 1) * ehat - 2) / ((J - 1) * (m1.shape[0] - 1 - (J - 1) * ehat))
    etil = min(1., etil)
    df1 = (J - 1) * etil
    df2 = (J - 1) * etil * (m1.shape[0] - 2 * g - 1)
    siglevel = 1 - f.cdf(test, df1, df2)
    
    result = {'test': test, 'df1': df1, 'df2': df2, 'p.value': siglevel, 'call': kwargs}
    result = pd.Series(result)
    result.name = 't1way'
    return result