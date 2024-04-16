import numpy as np
import pandas as pd
from scipy.stats import trim_mean

def t1waybt(formula, data, tr=0.2, nboot=599, *args, **kwargs):
    if data is None:
        mf = pd.DataFrame(formula)
    else:
        mf = pd.DataFrame(formula, data)
    cl = locals()
    x = mf.groupby(mf.columns[1])['response'].apply(list).tolist()
    if tr == 0.5:
        print("Comparing medians should not be done with this function!")
    grp = list(range(1, len(x)+1))
    J = len(x)
    for j in range(J):
        temp = x[j]
        x[j] = [val for val in temp if not pd.isna(val)]
    bvec = np.zeros((J, 2, nboot))
    hval = np.zeros(J)
    
    for j in range(J):
        hval[j] = len(x[grp[j]]) - 2 * np.floor(tr * len(x[grp[j]]))
        xcen = [val - np.mean(x[grp[j]], tr) for val in x[grp[j]]]
        data = np.random.choice(xcen, size=len(x[grp[j]]) * nboot, replace=True).reshape(nboot, len(x[grp[j]]))
        bvec[j, :, :] = np.apply_along_axis(trim_mean, 1, data, tr)
    
    m1 = bvec[:, 0, :]
    m2 = bvec[:, 1, :]
    wvec = 1 / m2
    uval = np.apply_along_axis(np.sum, 1, wvec)
    blob = wvec * m1
    xtil = np.apply_along_axis(np.sum, 1, blob) / uval
    blob1 = np.zeros((J, nboot))
    for j in range(J):
        blob1[j, :] = wvec[j, :] * (m1[j, :] - xtil) ** 2
    avec = np.apply_along_axis(np.sum, 1, blob1) / (len(x) - 1)
    blob2 = (1 - wvec / uval) ** 2 / (hval - 1)
    cvec = np.apply_along_axis(np.sum, 1, blob2)
    cvec = 2 * (len(x) - 2) * cvec / (len(x) ** 2 - 1)
    testb = avec / (cvec + 1)
    ct = np.sum(np.isnan(testb))
    if ct > 0:
        print("Some bootstrap estimates of the test statistic could not be computed.")
    neff = np.sum(~np.isnan(testb))
    test = t1wayv2(x, tr=tr, grp=grp)
    pval = np.mean(test['TEST'] <= testb, axis=0)
    result = {'test': test['TEST'], 'p.value': pval, 'Var.Explained': test['Var.Explained'], 
              'Effect.Size': test['Effect.Size'], 'nboot.eff': neff, 'call': cl}
    result = pd.DataFrame(result)
    result.__class__ = ['t1waybt']
    return result