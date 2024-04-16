import numpy as np

def mgvar(m, se=False, op=0, cov_fun=covmve, SEED=True):
    if op == 0:
        temp = apgdis(m, se=se)['distance']
    if op != 0:
        temp = out(m, cov_fun=cov_fun, plotit=False, SEED=SEED)['dis']
    flag = (temp != min(temp))
    temp2 = temp
    temp2[~flag] = max(temp)
    flag2 = (temp2 != min(temp2))
    flag[~flag2] = False
    varvec = np.nan
    while sum(flag) > 0:
        ic = 0
        chk = np.nan
        remi = np.nan
        for i in range(m.shape[0]):
            if flag[i]:
                ic += 1
                chk[ic] = gvar(np.vstack((m[~flag, :], m[i, :])))
                remi[ic] = i
        sor = np.argsort(chk)
        k = remi[sor[0]]
        varvec[k] = chk[sor[0]]
        flag[k] = False
    return varvec