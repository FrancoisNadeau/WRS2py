def pbcor(x, y = None, beta = 0.2, ci = False, nboot = 500, alpha = 0.05, *args, **kwargs):
    import numpy as np
    import pandas as pd
    from scipy.stats import t
    
    if y is None:
        y = x.iloc[:, 1]
        x = x.iloc[:, 0]
    
    if len(x) != len(y):
        raise ValueError("The vectors do not have equal lengths!")
    
    m1 = pd.concat([x, y], axis=1)
    m1 = m1.dropna()
    nval = m1.shape[0]
    x = m1.iloc[:, 0]
    y = m1.iloc[:, 1]
    
    temp = np.sort(np.abs(x - np.median(x)))
    omhatx = temp[int((1 - beta) * len(x))]
    temp = np.sort(np.abs(y - np.median(y)))
    omhaty = temp[int((1 - beta) * len(y))]
    a = (x - pbos(x, beta)) / omhatx
    b = (y - pbos(y, beta)) / omhaty
    a = np.where(a <= -1, -1, a)
    a = np.where(a >= 1, 1, a)
    b = np.where(b <= -1, -1, b)
    b = np.where(b >= 1, 1, b)
    pbcor = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))
    test = pbcor * np.sqrt((len(x) - 2) / (1 - pbcor**2))
    sig = 2 * (1 - t.cdf(np.abs(test), len(x) - 2))
    
    if ci:
        data = np.random.choice(len(y), size=len(y) * nboot, replace=True).reshape(nboot, len(y))
        bvec = np.apply_along_axis(lambda i: pbcor_bootstrap(x[i], y[i], beta), 1, data)
        ihi = int((1 - alpha/2) * nboot + 0.5)
        ilow = int((alpha/2) * nboot + 0.5)
        bsort = np.sort(bvec)
        corci = [bsort[ilow], bsort[ihi]]
    else:
        corci = None
    
    result = {
        "cor": pbcor,
        "test": test,
        "p.value": sig,
        "n": nval,
        "cor_ci": corci,
        "alpha": alpha,
        "call": None
    }
    return result

def pbos(x, beta):
    return np.percentile(x, 100 * beta)

def pbcor_bootstrap(x, y, beta):
    import numpy as np
    
    temp = np.sort(np.abs(x - np.median(x)))
    omhatx = temp[int((1 - beta) * len(x))]
    temp = np.sort(np.abs(y - np.median(y)))
    omhaty = temp[int((1 - beta) * len(y))]
    a = (x - pbos(x, beta)) / omhatx
    b = (y - pbos(y, beta)) / omhaty
    a = np.where(a <= -1, -1, a)
    a = np.where(a >= 1, 1, a)
    b = np.where(b <= -1, -1, b)
    b = np.where(b >= 1, 1, b)
    pbcor = np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2))
    return pbcor