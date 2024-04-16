import numpy as np
from scipy.stats import iqr, norm

def near(x, pt, fr=1):
    m = np.median(np.abs(x - np.median(x)))
    if m == 0:
        temp = idealf(x)
        m = (temp['qu'] - temp['ql']) / (norm.ppf(0.75) - norm.ppf(0.25))
    if m == 0:
        m = np.sqrt(winvar(x) / 0.4129)
    if m == 0:
        raise ValueError("All measures of dispersion are equal to 0")
    dis = np.abs(x - pt)
    dflag = dis <= fr * m
    return dflag