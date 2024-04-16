import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.stats import mahalanobis

def sppbi(formula, id, data, est="mom", nboot=500, *args, **kwargs):
    if data is None:
        mf = pd.DataFrame(formula)
    else:
        mf = pd.DataFrame(formula, data)
    
    est = est.lower()
    if est not in ["mom", "onestep", "median"]:
        raise ValueError("Invalid value for 'est'.")
    
    mf1 = pd.DataFrame(formula)
    m = [i for i, name in enumerate(mf1.columns) if name in ["formula", "data", "id"]]
    mf1 = mf1.iloc[:, [0] + m]
    mf1["drop.unused.levels"] = True
    mf1.iloc[:, 0] = pd.DataFrame(stats.model.frame)
    mf1 = mf1.apply(eval, axis=1)
    random1 = mf1.iloc[:, 1]
    depvar = mf.columns[0]
    
    if len(np.unique(random1)) == len(mf.iloc[:, 2].value_counts()):
        ranvar = mf.columns[2]
        fixvar = mf.columns[1]
    else:
        ranvar = mf.columns[1]
        fixvar = mf.columns[2]
    
    MC = False
    K = len(mf.iloc[:, ranvar].value_counts())
    J = len(mf.iloc[:, fixvar].value_counts())
    p = J * K
    grp = np.arange(1, p+1)
    est = globals()[est]
    fixsplit = mf.groupby(fixvar)[depvar].apply(list)
    indsplit = mf.groupby(fixvar)[ranvar].apply(list)
    dattemp = pd.DataFrame(list(map(list, zip(fixsplit, indsplit))), columns=["fixsplit", "indsplit"])
    data = pd.DataFrame(dattemp.apply(lambda x: [item for sublist in zip(x.fixsplit, x.indsplit) for item in sublist], axis=1))
    x = data.copy()
    JK = J * K
    MJ = (J**2 - J) / 2
    MK = (K**2 - K) / 2
    JMK = J * MK
    Jm = J - 1
    jp = 1 - K
    kv = 0
    kv2 = 0
    for j in range(J):
        jp += K
        xmat = np.empty((len(x.iloc[jp]), K))
        for k in range(K):
            kv += 1
            xmat[:, k] = x.iloc[kv]
        xmat = elimna(xmat)
        for k in range(K):
            kv2 += 1
            x.iloc[kv2] = xmat[:, k]
    xx = x.copy()
    
    nvec = np.empty(J)
    jp = 1 - K
    for j in range(J):
        jp += K
        nvec[j] = len(x.iloc[jp])
    
    bloc = np.empty((nboot, J))
    
    mvec = np.empty(JMK)
    it = 0
    for j in range(J):
        x = np.empty((nvec[j], MK))
        im = 0
        for k in range(K):
            for kk in range(K):
                if k < kk:
                    im += 1
                    kp = j * K + k - K
                    kpp = j * K + kk - K
                    x[:, im] = xx.iloc[kp] - xx.iloc[kpp]
                    it += 1
                    mvec[it] = est(x[:, im])
        data = np.random.choice(nvec[j], size=nvec[j] * nboot, replace=True).reshape((nboot, -1))
        bvec = np.empty((nboot, MK))
        for k in range(MK):
            temp = x[:, k]
            bvec[:, k] = np.apply_along_axis(rmanogsub, 1, data, temp, est)
        if j == 0:
            bloc = bvec
        else:
            bloc = np.hstack((bloc, bvec))
    
    MJMK = MJ * MK
    con = np.zeros((JMK, MJMK))
    cont = np.zeros((J, MJ))
    ic = 0
    for j in range(J):
        for jj in range(J):
            if j < jj:
                ic += 1
                cont[j, ic] = 1
                cont[jj, ic] = -1
    tempv = np.zeros((MK-1, MJ))
    con1 = np.vstack((cont[0], tempv))
    for j in range(1, J):
        con2 = np.vstack((cont[j], tempv))
        con1 = np.vstack((con1, con2))
    con = con1
    if MK > 1:
        for k in range(2, MK):
            con1 = push(con1)
            con = np.hstack((con, con1))
    bcon = con.T @ bloc.T
    tvec = con.T @ mvec
    tvec = tvec[:, 0]
    tempcen = np.apply_along_axis(np.mean, 1, bcon)
    vecz = np.zeros(con.shape[1])
    bcon = bcon.T
    temp = bcon.copy()
    for ib in range(temp.shape[0]):
        temp[ib] = temp[ib] - tempcen + tvec
    smat = np.cov(temp)
    
    chkrank = np.linalg.matrix_rank(smat)
    bcon = np.vstack((bcon, vecz))
    if chkrank == smat.shape[1]:
        dv = mahalanobis(bcon, tvec, smat)
    if chkrank < smat.shape[1]:
        smat = inv(smat)
        dv = mahalanobis(bcon, tvec, smat, inverted=True)
    bplus = nboot + 1
    sig_level = 1 - np.sum(dv[bplus] >= dv[:nboot]) / nboot
    
    tvec1 = pd.DataFrame({"Estimate": tvec})
    rancomb = np.apply_along_axis(lambda x: "-".join(x), 0, np.array(np.meshgrid(mf[ranvar].unique(), mf[ranvar].unique())).T.reshape(-1, 2))
    fnames = mf[fixvar].unique()
    fcomb = np.apply_along_axis(lambda x: "-".join(x), 0, np.array(np.meshgrid(fnames, fnames)).T.reshape(-1, 2))
    tnames = np.apply_along_axis(lambda x: "-".join(x), 0, np.array(np.meshgrid(rancomb, fcomb)).T.reshape(-1, 2))
    tvec1.index = tnames
    result = {"test": tvec1, "p.value": sig_level, "contrasts": con, "call": cl}
    result = pd.DataFrame(result)
    result.__class__ = ["spp"]
    return result