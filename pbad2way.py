import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.linalg import kron

def pbad2way(formula, data, est="mom", nboot=599, pro_dis=False, *args, **kwargs):
    if data is None:
        mf = pd.DataFrame(formula)
    else:
        mf = pd.DataFrame(formula, data)
    
    est = est.lower()
    est = est if est in ["mom", "onestep", "median"] else "mom"
    
    J = len(mf.iloc[:, 1].unique())
    K = len(mf.iloc[:, 2].unique())
    alpha = 0.05
    conall = True
    op = False
    MM = False
    grp = np.nan
    JK = J * K
    est = globals()[est]
    nfac = mf.groupby([mf.iloc[:, 1], mf.iloc[:, 2]]).size().unstack()
    nfac1 = nfac.loc[mf.iloc[:, 1].unique(), mf.iloc[:, 2].unique()]
    
    data = data.dropna(subset=mf.columns)
    data = data.sort_values(by=[mf.columns[1], mf.columns[2]])
    data["row"] = np.repeat(np.arange(1, nfac1.shape[0] + 1), nfac1.values.flatten())
    dataMelt = pd.melt(data, id_vars=["row", mf.columns[1], mf.columns[2]], value_vars=mf.columns[0])
    dataWide = dataMelt.pivot_table(index="row", columns=[mf.columns[1], mf.columns[2]], values="value")
    dataWide = dataWide.droplevel(0, axis=1)
    x = dataWide
    
    if isinstance(x, pd.DataFrame):
        x = x.applymap(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    
    x = x.applymap(lambda x: x[~np.isnan(x)] if isinstance(x, np.ndarray) else x)
    
    if not conall:
        ij = np.ones((1, J))
        ik = np.ones((1, K))
        jm1 = J - 1
        cj = np.eye(jm1, J)
        for i in range(jm1):
            cj[i, i + 1] = -1
        km1 = K - 1
        ck = np.eye(km1, K)
        for i in range(km1):
            ck[i, i + 1] = -1
        conA = np.transpose(kron(cj, ik))
        conB = np.transpose(kron(ij, ck))
        conAB = np.transpose(kron(cj, ck))
        conAB = np.transpose(kron(np.abs(cj), ck))
    
    if conall:
        temp = con2way(J, K)
        conA = temp["conA"]
        conB = temp["conB"]
        conAB = temp["conAB"]
    
    ncon = max(conA.shape[0], conB.shape[0], conAB.shape[0])
    
    if not np.isnan(grp[0]):
        xx = []
        for i in range(len(grp)):
            xx.append(x[grp[i]])
        x = xx
    
    mvec = np.nan * np.ones(JK)
    for j in range(JK):
        temp = x[j]
        temp = temp[~np.isnan(temp)]
        x[j] = temp
        mvec[j] = est(temp)
    
    bvec = np.nan * np.ones((JK, nboot))
    for j in range(JK):
        data = np.random.choice(x[j], size=len(x[j]) * nboot, replace=True).reshape(nboot, -1)
        bvec[j] = np.apply_along_axis(est, 1, data)
        naind = np.where(np.isnan(bvec[j]))[0]
        if len(naind) > 0:
            bvec[j, naind] = np.mean(bvec[j], axis=0, keepdims=True)
    
    bconA = np.transpose(conA) @ bvec
    tvecA = np.transpose(conA) @ mvec
    tvecA = tvecA[:, 0]
    tempcenA = np.apply_along_axis(np.mean, 1, bconA)
    veczA = np.zeros(conA.shape[1])
    bconA = np.transpose(bconA)
    smatA = np.cov(bconA - tempcenA + tvecA, rowvar=False)
    bconA = np.vstack((bconA, veczA))
    
    if not pro_dis:
        if not op:
            dv = mahalanobis(bconA, tvecA, smatA)
        if op:
            dv = out(bconA)["dis"]
    
    if pro_dis:
        dv = pdis(bconA, MM=MM)
    
    bplus = nboot + 1
    sig_levelA = 1 - np.sum(dv[bplus:] >= dv[:nboot]) / nboot
    
    bconB = np.transpose(conB) @ bvec
    tvecB = np.transpose(conB) @ mvec
    tvecB = tvecB[:, 0]
    tempcenB = np.apply_along_axis(np.mean, 1, bconB)
    veczB = np.zeros(conB.shape[1])
    bconB = np.transpose(bconB)
    smatB = np.cov(bconB - tempcenB + tvecB, rowvar=False)
    bconB = np.vstack((bconB, veczB))
    
    if not pro_dis:
        if not op:
            dv = mahalanobis(bconB, tvecB, smatB)
        if op:
            dv = out(bconA)["dis"]
    
    if pro_dis:
        dv = pdis(bconB, MM=MM)
    
    sig_levelB = 1 - np.sum(dv[bplus:] >= dv[:nboot]) / nboot
    
    bconAB = np.transpose(conAB) @ bvec
    tvecAB = np.transpose(conAB) @ mvec
    tvecAB = tvecAB[:, 0]
    tempcenAB = np.apply_along_axis(np.mean, 1, bconAB)
    veczAB = np.zeros(conAB.shape[1])
    bconAB = np.transpose(bconAB)
    smatAB = np.cov(bconAB - tempcenAB + tvecAB, rowvar=False)
    bconAB = np.vstack((bconAB, veczAB))
    
    if not pro_dis:
        if not op:
            dv = mahalanobis(bconAB, tvecAB, smatAB)
        if op:
            dv = out(bconAB)["dis"]
    
    if pro_dis:
        dv = pdis(bconAB, MM=MM)
    
    sig_levelAB = 1 - np.sum(dv[bplus:] >= dv[:nboot]) / nboot
    
    result = {
        "Qa": np.nan,
        "A.p.value": sig_levelA,
        "Qb": np.nan,
        "B.p.value": sig_levelB,
        "Qab": np.nan,
        "AB.p.value": sig_levelAB,
        "call": cl,
        "varnames": list(mf.columns),
        "dim": [J, K]
    }
    
    return result