import numpy as np
import pandas as pd

def rmanovab(y, groups, blocks, tr=0.2, nboot=599, **kwargs):
    cols1 = str(y)
    cols2 = str(groups)
    cols3 = str(blocks)
    dat = pd.DataFrame({cols1: y, cols2: groups, cols3: blocks})
    dat.columns = [cols1, cols2, cols3]
    cl = kwargs.get('cl', None)
    x = dat.pivot(index=cols3, columns=cols2, values=cols1).iloc[:, 1:]
    alpha = 0.05
    grp = 0
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(x, np.ndarray):
        if np.sum(grp) == 0:
            grp = np.arange(x.shape[1]) + 1
        mat = x[:, grp-1]
    mat = elimna(mat)
    J = mat.shape[1]
    connum = (J**2 - J) / 2
    bvec = np.zeros((connum, nboot))
    
    data = np.random.choice(mat.shape[0], size=mat.shape[0]*nboot, replace=True).reshape(nboot, -1)
    xcen = np.zeros_like(mat)
    for j in range(J):
        xcen[:, j] = mat[:, j] - np.mean(mat[:, j], tr)
    bvec = np.apply_along_axis(tsubrmanovab, 1, data, xcen, tr)
    
    icrit = round((1 - alpha) * nboot)
    bvec = np.sort(bvec)
    crit = bvec[icrit]
    test = rmanovatemp(mat, tr, grp)['test']
    result = {'test': test, 'crit': crit, 'call': cl}
    result = pd.Series(result)
    result = result.astype('rmanovab')
    return result