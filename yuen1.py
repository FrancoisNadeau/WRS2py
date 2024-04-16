import numpy as np
from scipy.stats import t

def yuen1(x, y=None, tr=0.2, alpha=0.05, **kwargs):
    if y is None:
        if isinstance(x, (np.ndarray, list)):
            y = x[1]
            x = x[0]
        elif isinstance(x, (np.matrix, pd.DataFrame)):
            y = x.iloc[:, 1]
            x = x.iloc[:, 0]
    
    if tr == 0.5:
        raise ValueError("Using tr=0.5 is not allowed; use a method designed for medians")
    if tr > 0.25:
        print("Warning: with tr > 0.25 type I error control might be poor")
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    h1 = len(x) - 2 * np.floor(tr * len(x))
    h2 = len(y) - 2 * np.floor(tr * len(y))
    
    q1 = (len(x) - 1) * winvar(x, tr) / (h1 * (h1 - 1))
    q2 = (len(y) - 1) * winvar(y, tr) / (h2 * (h2 - 1))
    
    df = (q1 + q2)**2 / ((q1**2 / (h1 - 1)) + (q2**2 / (h2 - 1)))
    crit = t.ppf(1 - alpha / 2, df)
    
    dif = np.mean(x, tr) - np.mean(y, tr)
    low = dif - crit * np.sqrt(q1 + q2)
    up = dif + crit * np.sqrt(q1 + q2)
    
    test = np.abs(dif / np.sqrt(q1 + q2))
    yuen = 2 * (1 - t.cdf(test, df))
    
    return {
        'n1': len(x),
        'n2': len(y),
        'est.1': np.mean(x, tr),
        'est.2': np.mean(y, tr),
        'ci': [low, up],
        'p.value': yuen,
        'dif': dif,
        'se': np.sqrt(q1 + q2),
        'teststat': test,
        'alpha': alpha,
        'crit': crit,
        'df': df
    }