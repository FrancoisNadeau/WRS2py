import numpy as np
from scipy.stats import binom

def dep_effect(x, y, tr=0.2, nboot=1000, *args, **kwargs):
    REL_MAG = None
    SEED = False
    ecom = np.array([0.10, 0.54, 0.54, 0.46, 0.30, 0.62, 0.62, 0.38, 0.50, 0.69, 0.69, 0.31])
    REL_EF = np.empty((4, 3))
    if y is not None:
        x = x - y
    x = elimna(x)
    n = len(x)
    output = np.empty((4, 7))
    output[:] = np.nan
    output[0, 0:2] = [0, D_akp_effect(x, tr=tr)]
    output[1, 0:2] = [0.5, depQS(x)['Q.effect']]
    output[2, 0:2] = [0.5, depQS(x, locfun=np.mean, tr=tr)['Q.effect']]
    output[3, 0:2] = [0.5, np.mean(x[x != 0] < 0)]
    if REL_MAG is None:
        REL_MAG = np.array([0.1, 0.3, 0.5])
        REL_EF = np.tile(ecom, (4, 1))
    if output[0, 1] < 0:
        REL_EF[0, :] = 0 - REL_EF[0, :]
    if output[1, 1] < 0.5:
        REL_EF[1, :] = 0.5 - (REL_EF[1, :] - 0.5)
    if output[2, 1] < 0.5:
        REL_EF[2, :] = 0.5 - (REL_EF[2, :] - 0.5)
    if output[3, 1] > 0.5:
        REL_EF[3, :] = 0.5 - (REL_EF[3, :] - 0.5)
    output[:, 2:5] = REL_EF
    output[0, 2:5] = REL_MAG
    output[0, 5:7] = D_akp_effect_ci(x, SEED=SEED, tr=tr, nboot=nboot)['ci']
    output[1, 5:7] = depQSci(x, SEED=SEED, nboot=nboot)['ci']
    output[2, 5:7] = depQSci(x, locfun=np.mean, SEED=SEED, tr=tr, nboot=nboot)['ci']
    Z = np.sum(x < 0)
    output[3, 5:7] = binom.conf(Z, n)['ci']
    output = output.astype(object)
    output.__class__ = ["matrix", "array", "dep.effect"]
    return output