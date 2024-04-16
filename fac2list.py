def fac2list(x, g):
    import pandas as pd
    import numpy as np
    
    g = pd.DataFrame(g)
    L = g.shape[1]
    g = g.apply(lambda col: col.astype('category'))
    g = g.apply(lambda col: col.cat.codes)
    g = g.values.tolist()
    
    for j in range(L):
        g[j] = pd.Series(g[j]).astype('category')
    
    Lp1 = L + 1
    if L > 4:
        raise ValueError("Can have at most 4 factors")
    
    if L == 1:
        res = selby(pd.concat([x, pd.DataFrame(g)], axis=1), 2, 1)
        group_id = res['grpn']
        res = res['x']
    
    if L > 1:
        res = selby2(pd.concat([x, pd.DataFrame(g)], axis=1), list(range(2, Lp1)), 1)
        group_id = res['grpn']
        res = res['x']
    
    return res