import numpy as np
from scipy.stats import norm

def winvarN(x, tr=0.2, ...):
    x = elimna(x)
    cterm = None
    if tr == 0:
        cterm = 1
    elif tr == 0.1:
        cterm = 0.6786546
    elif tr == 0.2:
        cterm = 0.4120867
    if cterm is None:
        cterm = norm.cdf(norm.ppf(tr)) + 2 * (norm.ppf(tr) ** 2) * tr
    bot = winvar(x, tr=tr) / cterm
    return bot