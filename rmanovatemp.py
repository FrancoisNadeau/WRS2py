import numpy as np
from scipy.stats import f

def rmanovatemp(x, tr=0.2, grp=None):
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    
    if isinstance(x, list):
        J = len(grp)
        m1 = np.matrix(x[grp[0]])
        for i in range(1, J):
            m2 = np.matrix(x[grp[i]])
            m1 = np.hstack((m1, m2))
    
    if isinstance(x, np.ndarray):
        if len(grp) < x.shape[1]:
            m1 = np.matrix(x[:, grp])
        else:
            m1 = np.matrix(x)
        J = x.shape[1]
    
    m2 = np.zeros_like(m1)
    xvec = np.ones(J)
    g = int(tr * m1.shape[0])
    
    for j in range(m1.shape[1]):
        m2[:, j] = winval(m1[:, j], tr)
        xvec[j] = np.mean(m1[:, j])
    
    xbar = np.mean(xvec)
    qc = (m1.shape[0] - 2 * g) * np.sum((xvec - xbar) ** 2)
    
    m3 = np.zeros_like(m1)
    m3 = m2 - np.mean(m2, axis=1).reshape(-1, 1)
    m3 = m3 - np.mean(m3, axis=0)
    m3 = m3 + np.mean(m2)
    qe = np.sum(m3 ** 2)
    
    test = qc / (qe / (m1.shape[0] - 2 * g - 1))
    
    v = winall(m1, tr=tr)['cov']
    vbar = np.mean(v)
    vbar_d = np.mean(np.diag(v))
    vbar_j = np.ones(J)
    
    for j in range(J):
        vbar_j[j] = np.mean(v[j, :])
    
    A = J * J * (vbar_d - vbar) ** 2 / (J - 1)
    B = np.sum(v * v) - 2 * J * np.sum(vbar_j ** 2) + J * J * vbar ** 2
    ehat = A / B
    
    etil = (m2.shape[0] * (J - 1) * ehat - 2) / ((J - 1) * (m2.shape[0] - 1 - (J - 1) * ehat))
    etil = min(1., etil)
    
    df1 = (J - 1) * etil
    df2 = (J - 1) * etil * (m2.shape[0] - 2 * g - 1)
    siglevel = 1 - f.cdf(test, df1, df2)
    
    return {'test': test, 'df': [df1, df2], 'siglevel': siglevel, 'tmeans': xvec, 'ehat': ehat, 'etil': etil}