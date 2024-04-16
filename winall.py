import numpy as np
import pandas as pd

def winall(x, tr=0.2, **kwargs):
    m = x
    cl = locals()
    if isinstance(m, pd.DataFrame):
        m = m.values
    if not isinstance(m, np.ndarray):
        raise ValueError("The data must be stored in a n by p matrix")
    wcor = np.ones((m.shape[1], m.shape[1]))
    wcov = np.zeros((m.shape[1], m.shape[1]))
    siglevel = np.empty((m.shape[1], m.shape[1]))
    siglevel[:] = np.nan
    for i in range(m.shape[1]):
        ip = i
        for j in range(ip, m.shape[1]):
            val = wincor(m[:, i], m[:, j], tr)
            wcor[i, j] = val['cor']
            wcor[j, i] = wcor[i, j]
            if i == j:
                wcor[i, j] = 1
            wcov[i, j] = val['cov']
            wcov[j, i] = wcov[i, j]
            if i != j:
                siglevel[i, j] = val['p.value']
                siglevel[j, i] = siglevel[i, j]
    if x.columns is not None:
        colnames = x.columns
        rownames = colnames
    else:
        colnames = np.arange(m.shape[1])
        rownames = colnames
    result = {'cor': wcor, 'cov': wcov, 'p.values': siglevel, 'call': cl}
    result = pd.DataFrame(result)
    result.columns = colnames
    result.index = rownames
    result.index.name = None
    result.columns.name = None
    result = result.astype(object)
    result.__class__.__name__ = 'pball'
    return result