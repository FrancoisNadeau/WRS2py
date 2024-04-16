import numpy as np

def rmdzero(x, est=onestep, grp=None, nboot=500, *args, **kwargs):
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    
    if isinstance(x, list):
        mat = np.zeros((len(x[0]), len(x)))
        for j in range(len(x)):
            mat[:, j] = x[j]
    
    if isinstance(x, np.ndarray):
        mat = x
    
    if grp is not None and not np.isnan(grp[0]):
        mat = mat[:, grp]
    
    mat = elimna(mat)
    J = mat.shape[1]
    jp = 0
    Jall = (J**2 - J) / 2
    dif = np.empty((mat.shape[0], Jall))
    ic = 0
    
    for j in range(J):
        for k in range(J):
            if j < k:
                ic += 1
                dif[:, ic] = mat[:, j] - mat[:, k]
    
    dif = np.asmatrix(dif)
    
    data = np.random.choice(mat.shape[0], size=mat.shape[0] * nboot, replace=True).reshape(nboot, mat.shape[0])
    bvec = np.empty((nboot, dif.shape[1]))
    
    for j in range(dif.shape[1]):
        temp = dif[:, j]
        bvec[:, j] = np.apply_along_axis(rmanogsub, 1, data, temp, est)
    
    center = np.apply_along_axis(est, 0, dif, *args, **kwargs)
    bcen = np.mean(bvec, axis=0)
    cmat = np.var(bvec - bcen + center, axis=0)
    zvec = np.zeros(Jall)
    m1 = np.vstack((bvec, zvec))
    bplus = nboot + 1
    discen = mahalanobis(m1, center, cmat)
    sig_level = np.sum(discen[bplus] <= discen[:nboot]) / nboot
    
    return {"discen": discen, "p.value": sig_level, "center": center}