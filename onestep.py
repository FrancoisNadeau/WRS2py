import numpy as np

def onestep(x, bend=1.28, na_rm=False, med=True, *args, **kwargs):
    if na_rm:
        x = x[~np.isnan(x)]
    if med:
        init_loc = np.median(x)
    else:
        init_loc = mom(x, bend=bend)
    y = (x - init_loc) / np.median(np.abs(x - np.median(x)))
    A = np.sum(hpsi(y, bend))
    B = len(x[np.abs(y) <= bend])
    onestep = np.median(x) + np.median(np.abs(x - np.median(x))) * A / B
    return onestep