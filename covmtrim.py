import numpy as np

def covmtrim(x, tr=0.2, p=len(x), grp=list(range(1, p+1))):
    if isinstance(x, list):
        x = matl(x)
    x = elimna(x)
    x = listm(x)
    if not isinstance(x, list):
        raise ValueError("The data are not stored in list mode or a matrix.")
    p = len(grp)
    pm1 = p - 1
    for i in range(pm1):
        ip = i + 1
        if len(x[grp[ip]]) != len(x[grp[i]]):
            raise ValueError("The number of observations in each group must be equal")
    n = len(x[grp[0]])
    h = len(x[grp[0]]) - 2 * int(tr * len(x[grp[0]]))
    covest = np.zeros((p, p))
    covest[0, 0] = (n - 1) * winvar(x[grp[0]], tr) / (h * (h - 1))
    for j in range(1, p):
        jk = j - 1
        covest[j, j] = (n - 1) * winvar(x[grp[j]], tr) / (h * (h - 1))
        for k in range(jk):
            covest[j, k] = (n - 1) * wincor(x[grp[j]], x[grp[k]], tr, ci=False).cov / (h * (h - 1))
            covest[k, j] = covest[j, k]
    covmtrim = covest
    return covmtrim