import numpy as np

def twocor(x1, y1, x2, y2, corfun="pbcor", nboot=599, tr=0.2, beta=0.2, **kwargs):
    alpha = 0.05
    corfun = corfun.lower()
    if corfun not in ["pbcor", "wincor"]:
        raise ValueError("Invalid value for corfun. Must be 'pbcor' or 'wincor'.")
    
    data1 = np.random.choice(len(y1), size=len(y1)*nboot, replace=True).reshape(nboot, -1)
    if corfun == "pbcor":
        bvec1 = np.apply_along_axis(lambda xx: pbcor(x1[xx], y1[xx], beta=beta, ci=False)["cor"], 1, data1)
    if corfun == "wincor":
        bvec1 = np.apply_along_axis(lambda xx: wincor(x1[xx], y1[xx], tr=tr, ci=False)["cor"], 1, data1)
    
    data2 = np.random.choice(len(y2), size=len(y2)*nboot, replace=True).reshape(nboot, -1)
    if corfun == "pbcor":
        bvec2 = np.apply_along_axis(lambda xx: pbcor(x2[xx], y2[xx], beta=beta, ci=False)["cor"], 1, data2)
    if corfun == "wincor":
        bvec2 = np.apply_along_axis(lambda xx: wincor(x2[xx], y2[xx], tr=tr, ci=False)["cor"], 1, data2)
    
    bvec = bvec1 - bvec2
    bsort = np.sort(bvec)
    nboot = len(bsort)
    term = alpha / 2
    ilow = round((alpha / 2) * nboot)
    ihi = nboot - ilow
    ilow += 1
    corci = [0, 0]
    corci[0] = bsort[ilow]
    corci[1] = bsort[ihi]
    
    pv = (np.sum(bsort < 0) + 0.5 * np.sum(bsort == 0)) / nboot
    pv = 2 * min(pv, 1 - pv)
    
    if corfun == "pbcor":
        r1 = pbcor(x1, y1, beta, ci=False)["cor"]
        r2 = pbcor(x2, y2, beta, ci=False)["cor"]
    if corfun == "wincor":
        r1 = wincor(x1, y1, tr, ci=False)["cor"]
        r2 = wincor(x2, y2, tr, ci=False)["cor"]
    
    result = {"r1": r1, "r2": r2, "ci": corci, "p.value": pv}
    result.update(kwargs)
    return result