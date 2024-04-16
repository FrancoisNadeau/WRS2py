import numpy as np
import scipy.stats as stats

def msmedse(x, sewarn=True, *args, **kwargs):
    x = elimna(x)
    chk = np.sum(np.duplicated(x))
    if sewarn:
        if chk > 0:
            print("Tied values detected. Estimate of standard error might be inaccurate.")
    y = np.sort(x)
    n = len(x)
    av = round((n + 1) / 2 - stats.norm.ppf(.995) * np.sqrt(n / 4))
    if av == 0:
        av = 1
    top = n - av + 1
    sqse = ((y[top] - y[av]) / (2 * stats.norm.ppf(.995))) ** 2
    sqse = np.sqrt(sqse)
    return sqse