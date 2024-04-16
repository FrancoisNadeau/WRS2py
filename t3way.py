import numpy as np
from scipy.stats import f

def t3way(formula, data, tr=0.2, *args):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    
    J = len(np.unique(mf[:, 1]))
    K = len(np.unique(mf[:, 2]))
    L = len(np.unique(mf[:, 3]))
    p = J * K * L
    grp = np.arange(1, p+1)
    lev_col = [1, 2, 3]
    var_col = 0
    alpha = 0.05
    
    data = np.array(mf)
    temp = selby2(data, lev_col, var_col)
    lev1 = len(np.unique(temp[:, 0]))
    lev2 = len(np.unique(temp[:, 1]))
    lev3 = len(np.unique(temp[:, 2]))
    gv = np.apply_along_axis(rank, 0, temp)
    gvad = 100 * gv[:, 0] + 10 * gv[:, 1] + gv[:, 2]
    grp = rank(gvad)
    J = lev1
    K = lev2
    L = lev3
    data = temp[:, 3:]
    
    if isinstance(data, np.ndarray):
        data = listm(data)
    
    data = [list(map(float, d)) for d in data]
    
    tmeans = np.zeros(p)
    h = np.zeros(p)
    v = np.zeros(p)
    
    if len(grp) != p:
        raise ValueError("Incomplete design! It needs to be full factorial!")
    
    for i in range(p):
        tmeans[i] = np.mean(data[grp[i]], tr)
        h[i] = len(data[grp[i]]) - 2 * np.floor(tr * len(data[grp[i]]))
        v[i] = (len(data[grp[i]]) - 1) * winvar(data[grp[i]], tr) / (h[i] * (h[i] - 1))
    
    v = np.diag(v)
    ij = np.tile(np.arange(1, J+1), (1, 1))
    ik = np.tile(np.arange(1, K+1), (1, 1))
    il = np.tile(np.arange(1, L+1), (1, 1))
    jm1 = J - 1
    cj = np.eye(jm1, J)
    for i in range(jm1):
        cj[i, i+1] = -1
    km1 = K - 1
    ck = np.eye(km1, K)
    for i in range(km1):
        ck[i, i+1] = -1
    lm1 = L - 1
    cl = np.eye(lm1, L)
    for i in range(lm1):
        cl[i, i+1] = -1
    alval = np.arange(1, 1000) / 1000
    
    cmat = np.kron(cj, np.kron(ik, il))
    Qa = johan(cmat, tmeans, v, h, alpha)
    A_p_value = t3pval(cmat, tmeans, v, h)
    
    cmat = np.kron(ij, np.kron(ck, il))
    Qb = johan(cmat, tmeans, v, h, alpha)
    B_p_value = t3pval(cmat, tmeans, v, h)
    
    cmat = np.kron(ij, np.kron(ik, cl))
    for i in range(999):
        irem = i
        Qc = johan(cmat, tmeans, v, h, alval[i])
        if Qc['teststat'] > Qc['crit']:
            break
    C_p_value = irem / 1000
    
    cmat = np.kron(cj, np.kron(ck, il))
    for i in range(999):
        irem = i
        Qab = johan(cmat, tmeans, v, h, alval[i])
        if Qab['teststat'] > Qab['crit']:
            break
    AB_p_value = irem / 1000
    
    cmat = np.kron(cj, np.kron(ik, cl))
    for i in range(999):
        irem = i
        Qac = johan(cmat, tmeans, v, h, alval[i])
        if Qac['teststat'] > Qac['crit']:
            break
    AC_p_value = irem / 1000
    
    cmat = np.kron(ij, np.kron(ck, cl))
    for i in range(999):
        irem = i
        Qbc = johan(cmat, tmeans, v, h, alval[i])
        if Qbc['teststat'] > Qbc['crit']:
            break
    BC_p_value = irem / 1000
    
    cmat = np.kron(cj, np.kron(ck, cl))
    for i in range(999):
        irem = i
        Qabc = johan(cmat, tmeans, v, h, alval[i])
        if Qabc['teststat'] > Qabc['crit']:
            break
    ABC_p_value = irem / 1000
    
    result = {
        'Qa': Qa['teststat'],
        'A_p_value': A_p_value,
        'Qb': Qb['teststat'],
        'B_p_value': B_p_value,
        'Qc': Qc['teststat'],
        'C_p_value': C_p_value,
        'Qab': Qab['teststat'],
        'AB_p_value': AB_p_value,
        'Qac': Qac['teststat'],
        'AC_p_value': AC_p_value,
        'Qbc': Qbc['teststat'],
        'BC_p_value': BC_p_value,
        'Qabc': Qabc['teststat'],
        'ABC_p_value': ABC_p_value,
        'call': mcl,
        'varnames': colnames(mf)
    }
    result['class'] = ['t3way']
    return result