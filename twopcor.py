def twopcor(x1, y1, x2, y2, nboot=599, *args, **kwargs):
    import numpy as np
    import warnings
    
    cl = locals()
    if nboot < 599:
        warnings.warn("It is unknown how to adjust the confidence interval when n1+n2 < 250.")
    
    X = np.column_stack((x1, y1))
    x1 = X[:, 0]
    y1 = X[:, 1]
    
    X = np.column_stack((x2, y2))
    x2 = X[:, 0]
    y2 = X[:, 1]
    
    data1 = np.random.choice(len(y1), size=len(y1)*nboot, replace=True).reshape(nboot, -1)
    bvec1 = np.apply_along_axis(lambda xx: np.corrcoef(x1[xx], y1[xx])[0, 1], 1, data1)
    
    data2 = np.random.choice(len(y2), size=len(y2)*nboot, replace=True).reshape(nboot, -1)
    bvec2 = np.apply_along_axis(lambda xx: np.corrcoef(x2[xx], y2[xx])[0, 1], 1, data2)
    
    bvec = bvec1 - bvec2
    
    ilow = 15
    ihi = 584
    if len(y1) + len(y2) < 250:
        ilow = 14
        ihi = 585
    if len(y1) + len(y2) < 180:
        ilow = 11
        ihi = 588
    if len(y1) + len(y2) < 80:
        ilow = 8
        ihi = 592
    if len(y1) + len(y2) < 40:
        ilow = 7
        ihi = 593
    
    bsort = np.sort(bvec)
    r1 = np.corrcoef(x1, y1)[0, 1]
    r2 = np.corrcoef(x2, y2)[0, 1]
    ci = [bsort[ilow], bsort[ihi]]
    
    result = {'r1': r1, 'r2': r2, 'ci': ci, 'call': cl}
    result['class'] = 'twocor'
    
    return result