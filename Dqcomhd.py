import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def Dqcomhd(x, y, q = np.arange(0.1, 1, 0.1), nboot = 1000, na_rm = True, **kwargs):
    alpha = 0.05
    if na_rm:
        xy = pd.DataFrame({'x': x, 'y': y}).dropna()
        x = xy['x']
        y = xy['y']
    pv = []
    output = np.full((len(q), 10), np.nan)
    output[:, 0] = q
    output[:, 1] = len(x.dropna())
    output[:, 2] = len(y.dropna())
    for i in range(len(q)):
        output[i, 3] = np.percentile(x, q[i] * 100)
        output[i, 4] = np.percentile(y, q[i] * 100)
        output[i, 5] = output[i, 3] - output[i, 4]
        if na_rm:
            temp = bootdpci(x, y, est=hd, q=q[i], dif=False, plotit=False, pr=False, nboot=nboot, alpha=alpha, SEED=False)
            output[i, 6] = temp['output'][0, 4]
            output[i, 7] = temp['output'][0, 5]
            output[i, 9] = temp['output'][0, 2]
        else:
            temp = rmmismcp(x, y, est=hd, q=q[i], plotit=False, pr=False, nboot=nboot, alpha=alpha, SEED=False)
            output[i, 6] = temp['output'][0, 5]
            output[i, 7] = temp['output'][0, 6]
            output[i, 9] = temp['output'][0, 3]
    temp = np.argsort(output[:, 9])[::-1]
    zvec = alpha / np.arange(1, len(q) + 1)
    output[temp, 8] = zvec
    output = pd.DataFrame(output, columns=["q", "n1", "n2", "est1", "est2", "est1-est.2", "ci.low", "ci.up", "p.crit", "p.value"])
    output['signif'] = "YES"
    for i in range(output.shape[0]):
        if output.iloc[temp[i], 9] > output.iloc[temp[i], 8]:
            output.loc[i, 'signif'] = "NO"
        if output.iloc[temp[i], 9] <= output.iloc[temp[i], 8]:
            break
    output = output.iloc[:, :-1]
    result = {'partable': output, 'call': cl}
    result = pd.DataFrame(result)
    result.__class__ = "robtab"
    return result