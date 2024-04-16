import numpy as np
from scipy.stats import norm

from dnormvar import dnormvar
from elimna import elimna
from tmean import tmean
from winvar import winvar
from utilities import area, cl

def effect(formula, data, EQVAR=True, tr=0.2, nboot=200, alpha=0.05, *args, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data=data)
    
    xy = mf.response.groupby(mf.iloc[:, 1])
    faclevels = xy.groups.keys()
    x = xy.get_group(faclevels[0])
    y = xy.get_group(faclevels[1])
    x = elimna(x)
    y = elimna(y)
    n1 = len(x)
    n2 = len(y)
    
    s1sq = winvar(x, tr=tr)
    s2sq = winvar(y, tr=tr)
    spsq = (n1-1)*s1sq + (n2-1)*s2sq
    sp = np.sqrt(spsq / (n1+n2-2))
    cterm = 1
    if tr > 0:
        cterm = area(dnormvar, norm.ppf(tr), norm.ppf(1-tr)) + 2*(norm.ppf(tr)**2)*tr
    cterm = np.sqrt(cterm)
    if EQVAR:
        dval = cterm * (tmean(x, tr) - tmean(y, tr)) / sp
    else:
        dval = cterm * (tmean(x, tr) - tmean(y, tr)) / np.sqrt(s1sq)
    
    be_f = np.empty(nboot)
    for i in range(nboot):
        X = np.random.choice(x, n1, replace=True)
        Y = np.random.choice(y, n2, replace=True)
        s1sq = winvar(X, tr=tr)
        s2sq = winvar(Y, tr=tr)
        spsq = (n1-1)*s1sq + (n2-1)*s2sq
        sp = np.sqrt(spsq / (n1+n2-2))
        cterm = 1
        if tr > 0:
            cterm = area(dnormvar, norm.ppf(tr), norm.ppf(1-tr)) + 2*(norm.ppf(tr)**2)*tr
        cterm = np.sqrt(cterm)
        if EQVAR:
            dval_b = cterm * (tmean(X, tr) - tmean(Y, tr)) / sp
        else:
            dval_b = cterm * (tmean(X, tr) - tmean(Y, tr)) / np.sqrt(s1sq)
        be_f[i] = dval_b
    
    L = alpha * nboot / 2
    U = nboot - L
    be_f = np.sort(be_f)
    ci = [be_f[L], be_f[U]]
    
    result = {'AKPeffect': dval, 'AKPci': ci, 'alpha': alpha, 'call': cl}
    result['class'] = 'AKP'
    return result
