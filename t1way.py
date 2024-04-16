import numpy as np
from scipy.stats import t, f

def t1way(formula, data, tr=0.2, alpha=0.05, nboot=100, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data=data)
    cl = formula.match.call()
    x = np.split(mf.response, mf[,2])
    if tr == 0.5:
        print("Comparing medians should not be done with this function!")
    grp = np.arange(1, len(x)+1)
    J = len(grp)
    h = np.zeros(J)
    w = np.zeros(J)
    xbar = np.zeros(J)
    nv = np.nan
    pts = []
    for j in range(J):
        xx = ~np.isnan(x[j])
        val = x[j]
        x[j] = val[xx]
        nv[j] = len(x[j])
        h[j] = len(x[grp[j]]) - 2 * np.floor(tr * len(x[grp[j]]))
        w[j] = h[j] * (h[j] - 1) / ((len(x[grp[j]]) - 1) * winvar(x[grp[j]], tr))
        if winvar(x[grp[j]], tr) == 0:
            raise ValueError("Standard error cannot be computed because of Winsorized variance of 0 (e.g., due to ties). Try to decrease the trimming level.")
        xbar[j] = np.mean(x[grp[j]], tr)
        val = elimna(val)
        pts.extend(val)
    u = np.sum(w)
    xtil = np.sum(w * xbar) / u
    A = np.sum(w * (xbar - xtil)**2) / (J - 1)
    B = 2 * (J - 2) * np.sum((1 - w / u)**2 / (h - 1)) / (J**2 - 1)
    TEST = A / (B + 1)
    nu1 = J - 1
    nu2 = 1 / (3 * np.sum((1 - w / u)**2 / (h - 1)) / (J**2 - 1))
    sig = 1 - f.cdf(TEST, nu1, nu2)
    
    chkn = np.var(nv)
    if chkn == 0:
        top = np.var(xbar)
        bot = winvarN(pts, tr=tr)
        e_pow = np.sqrt(top / bot)
    
    vals = np.zeros(nboot)
    N = np.min(nv)
    xdat = [[] for _ in range(J)]
    for i in range(nboot):
        for j in range(J):
            xdat[j] = np.random.choice(x[j], N, replace=True)
        
        vals[i] = t1wayv2(xdat, tr=tr, nboot=5, SEED=False).Effect.Size
    loc_fun = np.median
    if chkn != 0:
        e_pow = loc_fun(vals, nanrm=True)
    
    ilow = round((alpha / 2) * nboot)
    ihi = nboot - ilow
    ilow += 1
    val = np.sort(vals)
    ci = [val[ilow], val[ihi]]
    
    result = {
        "test": TEST,
        "df1": nu1,
        "df2": nu2,
        "p.value": sig,
        "effsize": e_pow,
        "effsize_ci": ci,
        "alpha": alpha,
        "call": cl
    }
    result = {"t1way": result}
    return result