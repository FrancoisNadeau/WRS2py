import numpy as np

def winse(x, tr=0.2, *args):
    x = elimna(x)
    n = len(x)
    h = n - 2 * np.floor(tr * n)
    top = (n - 1) * np.sqrt(winvar(x, tr=tr))
    bot = (h - 1) * np.sqrt(n)
    se = top / bot
    return se