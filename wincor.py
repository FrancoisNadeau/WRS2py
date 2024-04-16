def wincor(x, y = None, tr = 0.2, ci = False, nboot = 500, alpha = 0.05, *args, **kwargs):
    cl = locals()
    if y is None:
        y = x[:, 1]
        x = x[:, 0]
    sig = None
    if len(x) != len(y):
        raise ValueError("Lengths of vectors are not equal")
    m1 = np.column_stack((x, y))
    m1 = elimna(m1)
    nval = m1.shape[0]
    x = m1[:, 0]
    y = m1[:, 1]
    g = int(tr * len(x))
    xvec = winval(x, tr)
    yvec = winval(y, tr)
    wcor = np.corrcoef(xvec, yvec)[0, 1]
    wcov = np.cov(xvec, yvec)[0, 1]
    if np.sum(x == y) != len(x):
        test = wcor * np.sqrt((len(x) - 2) / (1. - wcor**2))
        sig = 2 * (1 - stats.t.cdf(np.abs(test), len(x) - 2 * g - 2))
    else:
        test = None
        sig = None

    if ci:
        data = np.random.choice(len(y), size=len(y) * nboot, replace=True).reshape(nboot, -1)
        bvec = np.apply_along_axis(lambda i: np.corrcoef(winval(x[i], tr), winval(y[i], tr))[0, 1], 1, data)
        ihi = int((1 - alpha / 2) * nboot + 0.5)
        ilow = int((alpha / 2) * nboot + 0.5)
        bsort = np.sort(bvec)
        corci = [bsort[ilow], bsort[ihi]]
    else:
        corci = None

    result = {'cor': wcor, 'cov': wcov, 'test': test, 'p.value': sig, 'n': nval, 'cor_ci': corci, 'alpha': alpha, 'call': cl}
    result = {'pbcor': result}
    return result