def onesampb(x, est="onestep", nboot=2000, nv=0, alpha=0.05, *args, **kwargs):
    est = est.lower()
    if est not in ["mom", "onestep", "median"]:
        raise ValueError("Invalid value for 'est'. Must be one of 'mom', 'onestep', or 'median'.")
    
    if est == "mom":
        est_func = mom
    elif est == "onestep":
        est_func = onestep
    elif est == "median":
        est_func = median
    
    null_value = None
    if null_value is not None:
        nv = null_value
    
    x = elimna(x)
    data = np.random.choice(x, size=len(x)*nboot, replace=True).reshape(nboot, -1)
    bvec = np.apply_along_axis(est_func, 1, data)
    bvec = np.sort(bvec)
    low = int((alpha/2) * nboot)
    up = nboot - low
    low += 1
    pv = np.mean(bvec > nv) + 0.5 * np.mean(bvec == nv)
    pv = 2 * min(pv, 1 - pv)
    estimate = est_func(x)
    result = {"ci": [bvec[low], bvec[up]], "n": len(x), "estimate": estimate, "p.value": pv, "alpha": alpha}
    return result