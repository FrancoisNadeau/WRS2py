import numpy as np
from scipy.stats import trim_mean
from scipy.stats import t

def trimcibt(x, nv=0, tr=0.2, alpha=0.05, nboot=200, *args, **kwargs):
    test = (trim_mean(x, tr) - nv) / WRS2.trimse(x, tr)
    data = np.random.choice(x, size=len(x) * nboot, replace=True).reshape(nboot, -1) - trim_mean(x, tr)
    top = np.apply_along_axis(lambda row: trim_mean(row, tr), axis=1, arr=data)
    bot = np.apply_along_axis(lambda row: WRS2.trimse(row, tr), axis=1, arr=data)
    tval = np.sort(np.abs(top / bot))
    icrit = round((1 - alpha) * nboot)
    ibot = round(alpha * nboot / 2)
    itop = nboot - ibot
    trimcibt = trim_mean(x, tr) - tval[icrit] * WRS2.trimse(x, tr)
    trimcibt[1] = trim_mean(x, tr) + tval[icrit] * WRS2.trimse(x, tr)
    p_value = np.sum(np.abs(test) <= np.abs(tval)) / nboot
    result = {
        'estimate': trim_mean(x, tr),
        'ci': trimcibt,
        'test.stat': test,
        'tr': tr,
        'p.value': p_value,
        'n': len(x),
        'alpha': alpha
    }
    result['class'] = 'trimcibt'
    return result