import numpy as np

def trimse(x, tr=0.2, na_rm=False, *args, **kwargs):
    if na_rm:
        x = x[~np.isnan(x)]
    trimse = np.sqrt(winvar(x, tr)) / ((1 - 2 * tr) * np.sqrt(len(x)))
    return trimse