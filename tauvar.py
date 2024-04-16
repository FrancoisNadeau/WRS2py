import numpy as np
from scipy.stats import norm

def tauvar(x, cval=3):
    x = elimna(x)
    s = norm.ppf(0.75) * np.median(np.abs(x))
    y = (x - tauloc(x)) / s
    cvec = np.repeat(cval, len(x))
    W = np.minimum(y**2, cvec**2)
    val = s**2 * np.sum(W) / len(x)
    return val