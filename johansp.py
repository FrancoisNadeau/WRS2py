import numpy as np
from scipy.stats import f

def johansp(cmat, vmean, vsqse, h, J, K):
    p = J * K
    yvec = np.matrix(vmean).reshape(len(vmean), 1)
    test = cmat @ vsqse @ cmat.T
    invc = np.linalg.inv(test)
    test = yvec.T @ cmat.T @ invc @ cmat @ yvec
    temp = np.zeros(J)
    klim = 1 - K
    kup = 0
    for j in range(J):
        klim += K
        kup += K
        Q = np.zeros((p, p))
        for k in range(klim, kup + 1):
            Q[k, k] = 1
        mtem = vsqse @ cmat.T @ invc @ cmat @ Q
        temp[j] = (np.sum(np.diag(mtem @ mtem)) + np.sum(np.diag(mtem))**2) / (h[j] - 1)
    A = 0.5 * np.sum(temp)
    df1 = cmat.shape[0]
    df2 = cmat.shape[0] * (cmat.shape[0] + 2) / (3 * A)
    cval = cmat.shape[0] + 2 * A - 6 * A / (cmat.shape[0] + 2)
    test = test / cval
    sig = 1 - f.cdf(test, df1, df2)
    return {'teststat': test[0, 0], 'siglevel': sig, 'df': [df1, df2]}

