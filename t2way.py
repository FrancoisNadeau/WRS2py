import numpy as np
import pandas as pd
from scipy.stats import rankdata

def t2way(formula, data, tr=0.2, **kwargs):
    if data is None:
        mf = pd.DataFrame(formula)
    else:
        mf = pd.DataFrame(formula, data)
    
    cl = locals()
    
    if any(mf.iloc[:, 1].value_counts() * mf.iloc[:, 2].value_counts() == 0):
        raise ValueError("Estimation not possible due to incomplete design.")
    
    J = len(mf.iloc[:, 1].unique())
    K = len(mf.iloc[:, 2].unique())
    p = J * K
    grp = np.arange(1, p+1)
    lev_col = [1, 2]
    var_col = 0
    
    if tr == 0.5:
        raise ValueError("For medians, use med2way if there are no ties.")
    
    x = mf.values
    temp = selby2(x, lev_col, var_col)
    
    lev1 = len(np.unique(temp[:, 0]))
    lev2 = len(np.unique(temp[:, 1]))
    gv = np.apply_along_axis(rankdata, 0, temp)
    gvad = 10 * gv[:, 0] + gv[:, 1]
    grp = rankdata(gvad)
    J = lev1
    K = lev2
    x = temp[:, 2:]
    x = np.apply_along_axis(pd.to_numeric, 0, x)
    tmeans = np.zeros(p)
    h = np.zeros(p)
    v = np.zeros(p)
    
    for i in range(p):
        x[grp[i]-1] = elimna(x[grp[i]-1])
        tmeans[i] = np.mean(x[grp[i]-1], tr)
        h[i] = len(x[grp[i]-1]) - 2 * np.floor(tr * len(x[grp[i]-1]))
        v[i] = (len(x[grp[i]-1]) - 1) * winvar(x[grp[i]-1], tr) / (h[i] * (h[i] - 1))
    
    v = np.diag(v)
    ij = np.tile(1, J)
    ik = np.tile(1, K)
    jm1 = J - 1
    cj = np.eye(jm1, J)
    cj[cj == 1] = -1
    km1 = K - 1
    ck = np.eye(km1, K)
    ck[ck == 1] = -1
    cmat = np.kron(cj, ik)
    alval = np.arange(1, 1000) / 1000
    
    for i in range(999):
        irem = i
        Qa = johan(cmat, tmeans, v, h, alval[i])
        if Qa['teststat'] > Qa['crit']:
            break
    
    A_p_value = irem / 1000
    cmat = np.kron(ij, ck)
    
    for i in range(999):
        irem = i
        Qb = johan(cmat, tmeans, v, h, alval[i])
        if Qb['teststat'] > Qb['crit']:
            break
    
    B_p_value = irem / 1000
    cmat = np.kron(cj, ck)
    
    for i in range(999):
        irem = i
        Qab = johan(cmat, tmeans, v, h, alval[i])
        if Qab['teststat'] > Qab['crit']:
            break
    
    AB_p_value = irem / 1000
    tmeans = tmeans.reshape(J, K, order='F')
    result = {'Qa': Qa['teststat'], 'A_p_value': A_p_value, 'Qb': Qb['teststat'], 'B_p_value': B_p_value,
              'Qab': Qab['teststat'], 'AB_p_value': AB_p_value, 'call': cl, 'varnames': list(mf.columns),
              'dim': [J, K]}
    result['class'] = ['t2way']
    return result