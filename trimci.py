import numpy as np
from scipy.stats import t

def trimci(x, tr=0.2, alpha=0.05, null_value=0, pr=True, *args, **kwargs):
    x = elimna(x)
    se = np.sqrt(winvar(x, tr)) / ((1 - 2 * tr) * np.sqrt(len(x)))
    trimci = np.zeros(2)
    df = len(x) - 2 * np.floor(tr * len(x)) - 1
    trimci[0] = np.mean(x, tr) - t.ppf(1 - alpha / 2, df) * se
    trimci[1] = np.mean(x, tr) + t.ppf(1 - alpha / 2, df) * se
    test = (np.mean(x, tr) - null_value) / se
    sig = 2 * (1 - t.cdf(np.abs(test), df))
    return {'ci': trimci, 'estimate': np.mean(x, tr), 'test.stat': test, 'se': se, 'alpha': alpha, 'p.value': sig, 'n': len(x)}