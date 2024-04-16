import numpy as np
from scipy.stats import f

def t1wayv2(x, tr=0.2, grp=None, MAT=False, lev_col=1, var_col=2, nboot=100, SEED=False, pr=True, IV=None, loc_fun=np.median):
    if SEED:
        np.random.seed(2)
    
    if MAT:
        if not isinstance(x, np.ndarray):
            raise ValueError("With MAT=True, data must be stored in a matrix")
        if len(lev_col) != 1:
            raise ValueError("Argument lev_col should have 1 value")
        temp = selby(x, lev_col, var_col)
        x = temp['x']
        grp2 = np.argsort(temp['grpn'])
        x = x[grp2]
    
    if IV is not None and IV[0] is not None:
        if pr:
            print("Assuming x is a vector containing all of the data, the dependent variable")
        xi = elimna(np.column_stack((x, IV)))
        x = fac2list(xi[:, 0], xi[:, 1])
    
    if isinstance(x, np.ndarray):
        x = listm(x)
    
    if np.isnan(np.sum(grp[0])):
        grp = np.arange(1, len(x) + 1)
    
    if not isinstance(x, list):
        raise ValueError("Data are not stored in a matrix or in list mode.")
    
    J = len(grp)
    h = np.zeros(J)
    w = np.zeros(J)
    xbar = np.zeros(J)
    pts = None
    nval = np.zeros(J)
    
    for j in range(J):
        x[j] = elimna(x[j])
        val = x[j]
        val = elimna(val)
        nval[j] = len(val)
        pts = np.concatenate((pts, val))
        x[j] = val
        h[j] = len(x[grp[j]]) - 2 * np.floor(tr * len(x[grp[j]]))
        w[j] = h[j] * (h[j] - 1) / ((len(x[grp[j]]) - 1) * winvar(x[grp[j]], tr))
        xbar[j] = np.mean(x[grp[j]], tr)
    
    u = np.sum(w)
    xtil = np.sum(w * xbar) / u
    A = np.sum(w * (xbar - xtil)**2) / (J - 1)
    B = 2 * (J - 2) * np.sum((1 - w / u)**2 / (h - 1)) / (J**2 - 1)
    TEST = A / (B + 1)
    nu1 = J - 1
    nu2 = 1. / (3 * np.sum((1 - w / u)**2 / (h - 1)) / (J**2 - 1))
    sig = 1 - f.cdf(TEST, nu1, nu2)
    nv = [len(x[i]) for i in range(len(x))]
    
    chkn = np.var(nval)
    if chkn == 0:
        top = np.var(xbar)
        bot = winvarN(pts, tr=tr)
        e_pow = top / bot
    
    if chkn != 0:
        vals = np.zeros(nboot)
        N = min(nval)
        xdat = [[] for _ in range(J)]
        for i in range(nboot):
            for j in range(J):
                xdat[j] = np.random.choice(x[j], N)
                vals[i] = t1way.effect(xdat, tr=tr)['Var.Explained']
        e_pow = loc_fun(vals, nan_policy='omit')
    
    return {'TEST': TEST, 'nu1': nu1, 'nu2': nu2, 'n': nv, 'p.value': sig, 'Var.Explained': e_pow, 'Effect.Size': np.sqrt(e_pow)}