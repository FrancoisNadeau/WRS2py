import numpy as np
from scipy.stats import tmean
from scipy.linalg import sqrtm
from sklearn.covariance import MinCovDet

def pbadepth1(x, est=tmean, con=0, alpha=0.05, nboot=2000, grp=None, op=1, allp=True,
              MM=False, MC=False, cop=3, SEED=True, na_rm=False, *args, **kwargs):
    
    con = np.array(con)
    if isinstance(x, (np.ndarray, pd.DataFrame)):
        x = list(x)
    if not isinstance(x, list):
        raise ValueError("Data must be stored in list mode or in matrix mode.")
    if grp is not None:
        xx = []
        for i in grp:
            xx.append(x[i])
        x = xx
    J = len(x)
    mvec = np.empty(J)
    nvec = np.empty(J)
    for j in range(J):
        temp = x[j]
        if na_rm:
            temp = temp[~np.isnan(temp)]
        x[j] = temp
        mvec[j] = est(temp, *args, **kwargs)
        nvec[j] = len(temp)
    Jm = J - 1
    d = (J**2 - J) / 2 if con == 0 else con.shape[1]
    if np.sum(con**2) == 0:
        if allp:
            con = np.zeros((J, d))
            id = 0
            for j in range(Jm):
                jp = j + 1
                for k in range(jp, J):
                    id += 1
                    con[j, id] = 1
                    con[k, id] = -1
        else:
            con = np.zeros((J, Jm))
            for j in range(Jm):
                jp = j + 1
                con[j, j] = 1
                con[jp, j] = -1
    bvec = np.empty((J, nboot))
    if SEED:
        np.random.seed(2)
    for j in range(J):
        data = np.random.choice(x[j], size=len(x[j])*nboot, replace=True).reshape(nboot, -1)
        bvec[j] = np.apply_along_axis(est, 1, data, na_rm=na_rm, *args, **kwargs)
    chkna = np.sum(np.isnan(bvec))
    if chkna > 0:
        print("Bootstrap estimates of location could not be computed")
        print("This can occur when using an M-estimator")
        print("Might try est=tmean")
    bcon = con.T @ bvec
    tvec = con.T @ mvec
    tvec = tvec[:, 0]
    tempcen = np.apply_along_axis(np.mean, 1, bcon)
    vecz = np.zeros(con.shape[1])
    bcon = np.vstack((bcon, vecz))
    if op == 1:
        smat = np.cov(bcon - tempcen + tvec, rowvar=False)
        dv = mahalanobis(bcon, tvec, smat)
    if op == 2:
        smat = MinCovDet().fit(bcon - tempcen + tvec).covariance_
        dv = mahalanobis(bcon, tvec, smat)
    if op == 3:
        if not MC:
            dv = pdis(bcon, MM=MM, cop=cop)
    bplus = nboot + 1
    sig_level = 1 - np.sum(dv[bplus] >= dv[:nboot]) / nboot
    return {"p.value": sig_level, "psihat": tvec, "con": con, "n": nvec}