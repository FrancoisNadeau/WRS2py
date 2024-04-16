import numpy as np

def pb2gen(formula, data, est="mom", nboot=599, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data)
    
    cl = formula.match.call()
    xy = np.split(mf.model.extract("response"), mf[:, 2])
    faclevels = list(xy.keys())
    x = xy[0]
    y = xy[1]
    est = globals()[est]
    alpha = 0.05
    pr = True
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    datax = np.random.choice(x, size=len(x)*nboot, replace=True).reshape(nboot, -1)
    datay = np.random.choice(y, size=len(y)*nboot, replace=True).reshape(nboot, -1)
    bvecx = np.apply_along_axis(est, 1, datax)
    bvecy = np.apply_along_axis(est, 1, datay)
    bvec = np.sort(bvecx - bvecy)
    low = round((alpha/2)*nboot) + 1
    up = nboot - low
    temp = np.sum(bvec < 0)/nboot + np.sum(bvec == 0)/(2*nboot)
    sig_level = 2 * (min(temp, 1 - temp))
    se = np.var(bvec)
    result = {"test": est(x) - est(y), "conf.int": [bvec[low], bvec[up]], "p.value": sig_level, "call": cl}
    result["class"] = "pb2"
    return result