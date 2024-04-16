import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def bwtrim(formula, id, data, tr=0.2, *args):
    if data is None:
        mf = pd.DataFrame(formula)
    else:
        mf = pd.DataFrame(formula, data)
    
    cl = locals()
    mf1 = locals()
    m = [i for i, x in enumerate(mf1) if x in ["formula", "data", "id"]]
    mf1 = [mf1[i] for i in [0] + m]
    mf1["drop.unused.levels"] = True
    mf1[0] = "stats::model.frame"
    mf1 = eval(mf1, globals())
    random1 = mf1[id]
    depvar = mf.columns[0]
    
    if len(np.unique(random1)) == len(mf.iloc[:, 2].value_counts()):
        ranvar = mf.columns[2]
        fixvar = mf.columns[1]
    else:
        ranvar = mf.columns[1]
        fixvar = mf.columns[2]
    
    K = len(mf.iloc[:, 2].value_counts())
    J = len(mf.iloc[:, 1].value_counts())
    p = J * K
    grp = np.arange(1, p+1)
    fixsplit = mf.groupby(fixvar)[depvar].apply(list)
    indsplit = mf.groupby(fixvar)[ranvar].apply(list)
    dattemp = [pd.DataFrame({fixvar: fixsplit[i], ranvar: indsplit[i]}) for i in fixsplit.index]
    data = pd.concat(dattemp)
    tmeans = np.zeros(p)
    h = np.zeros(J)
    v = np.zeros((p, p))
    klow = 1 - K
    kup = 0
    
    for i in range(p):
        tmeans[i] = trim_mean(data.loc[grp[i], depvar], tr, na_rm=True)
    
    for j in range(J):
        h[j] = len(data.loc[grp[j], depvar]) - 2 * np.floor(tr * len(data.loc[grp[j], depvar]))
        klow += K
        kup += K
        sel = np.arange(klow, kup+1)
        v[sel, sel] = np.cov(data.loc[grp[klow:kup], :].T)
    
    ij = np.ones((1, J))
    ik = np.ones((1, K))
    jm1 = J - 1
    cj = np.eye(jm1, J)
    
    for i in range(jm1):
        cj[i, i+1] = -1
    
    km1 = K - 1
    ck = np.eye(km1, K)
    
    for i in range(km1):
        ck[i, i+1] = -1
    
    cmat = np.kron(cj, ik)
    Qa = johansp(cmat, tmeans, v, h, J, K)
    
    cmat = np.kron(ij, ck)
    Qb = johansp(cmat, tmeans, v, h, J, K)
    
    cmat = np.kron(cj, ck)
    Qab = johansp(cmat, tmeans, v, h, J, K)
    
    result = {"Qa": Qa["teststat"], "A.p.value": float(Qa["siglevel"]), "A.df": Qa["df"],
              "Qb": Qb["teststat"], "B.p.value": float(Qb["siglevel"]), "B.df": Qb["df"],
              "Qab": Qab["teststat"], "AB.p.value": float(Qab["siglevel"]), "AB.df": Qab["df"],
              "call": cl, "varnames": [depvar, fixvar, ranvar]}
    result = pd.DataFrame(result)
    result = result.astype({"A.p.value": float, "B.p.value": float, "AB.p.value": float})
    result = result.astype({"A.df": int, "B.df": int, "AB.df": int})
    result = result.astype({"Qa": float, "Qb": float, "Qab": float})
    result = result.astype({"call": str, "varnames": list})
    result = result.astype({"call": "bwtrim", "varnames": "bwtrim"})
    
    return result

tsplit = bwtrim
