import numpy as np
from scipy.stats import f, chi2

def med2way(formula, data, alpha=0.05):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    J = len(np.unique(mf[:, 2]))
    K = len(np.unique(mf[:, 3]))
    p = J * K
    grp = np.arange(1, p+1)
    
    x = np.array(mf)
    lev_col = [2, 3]
    var_col = 1
    temp = selby2(x, lev_col, var_col)
    x = [np.array(xi).astype(float) for xi in temp['x']]
    if p != len(x):
        print("Warning: The number of groups in your data is not equal to JK")
    
    xbar = np.zeros(p)
    h = np.zeros(p)
    d = np.zeros(p)
    R = np.zeros(J)
    W = np.zeros(K)
    nuhat = np.zeros(J)
    omegahat = np.zeros(K)
    DROW = np.zeros(J)
    DCOL = np.zeros(K)
    xtil = np.zeros((J, K))
    aval = np.zeros((J, K))
    
    for j in range(p):
        xbar[j] = np.median(x[grp[j]-1])
        h[j] = len(x[grp[j]-1])
        d[j] = msmedse(x[grp[j]-1], sewarn=False)**2
    
    d = d.reshape((J, K), order='F')
    xbar = xbar.reshape((J, K), order='F')
    h = h.reshape((J, K), order='F')
    
    for j in range(J):
        R[j] = np.sum(xbar[j, :])
        nuhat[j] = (np.sum(d[j, :]))**2 / np.sum(d[j, :]**2 / (h[j, :]-1))
        DROW[j] = np.sum(1 / d[j, :])
    
    for k in range(K):
        W[k] = np.sum(xbar[:, k])
        omegahat[k] = (np.sum(d[:, k]))**2 / np.sum(d[:, k]**2 / (h[:, k]-1))
        DCOL[k] = np.sum(1 / d[:, k])
    
    D = 1 / d
    
    for j in range(J):
        for k in range(K):
            xtil[j, k] = np.sum(D[:, k] * xbar[:, k] / DCOL[k]) + np.sum(D[j, :] * xbar[j, :] / DROW[j]) - np.sum(D * xbar / np.sum(D))
            aval[j, k] = (1 - D[j, k] * (1 / np.sum(D[j, :]) + 1 / np.sum(D[:, k]) - 1 / np.sum(D)))**2 / (h[j, k] - 3)
    
    Rhat = np.sum(r * R) / np.sum(r)
    What = np.sum(w * W) / np.sum(w)
    Ba = np.sum((1 - r / np.sum(r))**2 / nuhat)
    Bb = np.sum((1 - w / np.sum(w))**2 / omegahat)
    Va = np.sum(r * (R - Rhat)**2) / ((J - 1) * (1 + 2 * (J - 2) * Ba / (J**2 - 1)))
    Vb = np.sum(w * (W - What)**2) / ((K - 1) * (1 + 2 * (K - 2) * Bb / (K**2 - 1)))
    sig_A = 1 - f.cdf(Va, J - 1, 9999999)
    sig_B = 1 - f.cdf(Vb, K - 1, 9999999)
    
    Vab = np.sum(D * (xbar - xtil)**2)
    dfinter = (J - 1) * (K - 1)
    sig_AB = 1 - chi2.cdf(Vab, dfinter)
    
    result = {'Qa': Va, 'A.p.value': sig_A, 'Qb': Vb, 'B.p.value': sig_B, 'Qab': Vab, 'AB.p.value': sig_AB, 'call': cl,
              'varnames': colnames(mf), 'dim': [J, K]}
    result['class'] = ['t2way']
    return result