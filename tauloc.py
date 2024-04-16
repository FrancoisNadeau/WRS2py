import numpy as np
from scipy.stats import norm

def tauloc(x, cval=4.5):
    x = elimna(x)
    s = norm.ppf(0.75) * np.median(np.abs(x - np.median(x)))
    y = (x - np.median(x)) / s
    W = (1 - (y / cval) ** 2) ** 2
    flag = np.abs(W) > cval
    W[flag] = 0
    val = np.sum(W * x) / np.sum(W)
    return val