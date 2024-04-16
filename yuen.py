import numpy as np
from scipy.stats import t, ttest_ind

def yuen(formula, data=None, tr=0.2, *args, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data)
    cl = formula.match.call()
    xy = np.split(mf.response, mf.iloc[:, 1])
    faclevels = list(xy.keys())
    x = xy[faclevels[0]]
    y = xy[faclevels[1]]
    if tr == 0.5:
        print("Warning: Comparing medians should not be done with this function!")
    alpha = 0.05
    if y is None:
        if isinstance(x, (np.ndarray, pd.DataFrame)):
            y = x[:, 1]
            x = x[:, 0]
        if isinstance(x, list):
            y = x[1]
            x = x[0]
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
    es = np.abs(yuenv2(x, y, tr=tr).Effect.Size)
    result = {'test': test, 'conf.int': [low, up], 'p.value': yuen, 'df': df, 'diff': dif, 'effsize': es, 'call': cl}
    result['class'] = 'yuen'
    return result