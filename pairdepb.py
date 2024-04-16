import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def pairdepb(y, groups, blocks, tr=0.2, nboot=599, *args, **kwargs):
    cols1 = 'y'
    cols2 = 'groups'
    cols3 = 'blocks'
    dat = pd.DataFrame({'y': y, 'groups': groups, 'blocks': blocks})
    dat.columns = [cols1, cols2, cols3]
    x = dat.pivot(index=cols3, columns=cols2, values=cols1).values[:, 1:]
    grp = np.arange(1, x.shape[1]+1)
    alpha = 0.05
    if isinstance(x, pd.DataFrame):
        x = x.values
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    if isinstance(x, list):
        if np.sum(grp) == 0:
            grp = np.arange(1, len(x)+1)
        mat = np.zeros((len(x[0]), len(grp)))
        for j in range(len(grp)):
            mat[:, j] = x[grp[j]-1]
    if isinstance(x, np.ndarray):
        if np.sum(grp) == 0:
            grp = np.arange(1, x.shape[1]+1)
        mat = x[:, grp-1]
    if np.sum(np.isnan(mat)) >= 1:
        raise ValueError("Missing values are not allowed.")
    J = mat.shape[1]
    connum = (J**2 - J) / 2
    bvec = np.zeros((connum, nboot))
    
    data = np.random.choice(mat.shape[0], size=mat.shape[0]*nboot, replace=True).reshape(nboot, -1)
    xcen = np.zeros_like(mat)
    for j in range(J):
        xcen[:, j] = mat[:, j] - np.mean(mat[:, j], tr)
    it = 0
    for j in range(J):
        for k in range(J):
            if j < k:
                it += 1
                bvec[it-1, :] = np.apply_along_axis(lambda x: tsub(x, xcen[:, j], xcen[:, k], tr), 1, data)
    
    bvec = np.abs(bvec)
    icrit = round((1 - alpha) * nboot)
    critvec = np.apply_along_axis(np.max, 0, bvec)
    critvec = np.sort(critvec)
    crit = critvec[icrit-1]
    psihat = np.zeros((connum, 5))
    psihat[:, :2] = grp.reshape(-1, 1)
    psihat[:, 2] = np.nan
    psihat[:, 3] = np.nan
    psihat[:, 4] = np.nan
    test = np.zeros((connum, 4))
    test[:, :2] = grp.reshape(-1, 1)
    test[:, 2] = np.nan
    test[:, 3] = np.nan
    it = 0
    for j in range(J):
        for k in range(J):
            if j < k:
                it += 1
                estse = yuend(mat[:, j], mat[:, k])['se']
                dif = np.mean(mat[:, j], tr) - np.mean(mat[:, k], tr)
                psihat[it-1, 2] = dif
                psihat[it-1, 3] = dif - crit * estse
                psihat[it-1, 4] = dif + crit * estse
                test[it-1, 2] = yuend(mat[:, j], mat[:, k])['test']
                test[it-1, 3] = estse
    
    fnames = np.unique(groups).tolist()
    psihat1 = np.hstack((psihat, test[:, 2].reshape(-1, 1), crit))
    result = {'comp': psihat1, 'fnames': fnames, 'call': None}
    result['call'] = result
    return result

def tsub(data, x, y, tr):
    return np.mean(x[data], tr) - np.mean(y[data], tr)

def yuend(x, y):
    return ttest_ind(x, y)

class mcp1:
    pass