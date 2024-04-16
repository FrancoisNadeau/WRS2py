import numpy as np

def trimparts(x, tr=0.2):
    tm = np.mean(x, tr)
    h1 = len(x) - 2 * np.floor(tr * len(x))
    sqse = (len(x) - 1) * winvar(x, tr) / (h1 * (h1 - 1))
    trimparts = [tm, sqse]
    return trimparts