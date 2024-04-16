import numpy as np

def yuenbt(formula, data, tr=0.2, nboot=599, side=True, *args, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data)
    
    xy = np.split(mf.response, mf.iloc[:, 1])
    faclevels = list(xy.keys())
    x = xy[faclevels[0]]
    y = xy[faclevels[1]]
    alpha = 0.05
    nullval = 0
    pr = True
    plotit = False
    op = 1
    side = bool(side)
    p_value = np.nan
    yuenbt = np.zeros(2)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    xcen = x - np.mean(x, tr)
    ycen = y - np.mean(y, tr)
    test = (np.mean(x, tr) - np.mean(y, tr)) / np.sqrt(trimse(x, tr=tr)**2 + trimse(y, tr=tr)**2)
    datax = np.random.choice(xcen, size=len(x)*nboot, replace=True).reshape(nboot, -1)
    datay = np.random.choice(ycen, size=len(y)*nboot, replace=True).reshape(nboot, -1)
    top = np.apply_along_axis(np.mean, 1, datax, tr) - np.apply_along_axis(np.mean, 1, datay, tr)
    botx = np.apply_along_axis(trimse, 1, datax, tr)
    boty = np.apply_along_axis(trimse, 1, datay, tr)
    tval = top / np.sqrt(botx**2 + boty**2)
    if side:
        tval = np.abs(tval)
    tval = np.sort(tval)
    icrit = int((1 - alpha) * nboot + 0.5)
    ibot = int(alpha * nboot / 2 + 0.5)
    itop = int((1 - alpha / 2) * nboot + 0.5)
    se = np.sqrt(trimse(x, tr)**2 + trimse(y, tr)**2)
    yuenbt[0] = np.mean(x, tr) - np.mean(y, tr) - tval[itop] * se
    yuenbt[1] = np.mean(x, tr) - np.mean(y, tr) - tval[ibot] * se
    if side:
        yuenbt[0] = np.mean(x, tr) - np.mean(y, tr) - tval[icrit] * se
        yuenbt[1] = np.mean(x, tr) - np.mean(y, tr) + tval[icrit] * se
        p_value = np.sum(np.abs(test) <= np.abs(tval)) / nboot
    mdiff = np.mean(x, tr) - np.mean(y, tr)
    result = {'test': test, 'conf.int': yuenbt, 'p.value': p_value, 'df': np.nan, 'diff': mdiff, 'call': cl}
    result = result
    return result

def trimse(x, tr):
    return np.sqrt(np.mean((x - np.mean(x, tr))**2))

class yuen:
    pass