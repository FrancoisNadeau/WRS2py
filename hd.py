import numpy as np
from scipy.stats import beta

def hd(x, q=0.5, na_rm=True, stand=None):
    if na_rm:
        x = elimna(x)
    n = len(x)
    m1 = (n + 1) * q
    m2 = (n + 1) * (1 - q)
    vec = np.arange(1, n+1)
    w = beta.cdf(vec/n, m1, m2) - beta.cdf((vec-1)/n, m1, m2)
    y = np.sort(x)
    hd = np.sum(w * y)
    return hd