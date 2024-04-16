import numpy as np
from scipy.optimize import minimize

def spat(x):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a matrix")
    x = elimna(x)
    START = np.apply_along_axis(np.median, 0, x)
    val = minimize(spat.sub, START, args=(x,), method='BFGS').x
    return val