def yuen_effect_ci(formula, data, tr=0.2, nboot=400, alpha=0.05, *args, **kwargs):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    xy = split(model.extract(mf, "response"), mf[,2])
    faclevels = names(xy)
    x = xy[[1]]
    y = xy[[2]]
    x = elimna(x)
    y = elimna(y)
    bvec = [0] * nboot
    datax = np.random.choice(x, size=len(x)*nboot, replace=True).reshape(nboot, -1)
    datay = np.random.choice(y, size=len(x)*nboot, replace=True).reshape(nboot, -1)
    for i in range(nboot):
        bvec[i] = yuenv2(datax[i,], datay[i,], tr=tr, SEED=False)['Effect.Size']
    bvec = sorted(abs(bvec))
    crit = alpha / 2
    icl = round(crit * nboot) + 1
    icu = nboot - icl
    ci = [None, None]
    ci[0] = bvec[icl]
    pchk = yuen(formula=formula, data=mf, tr=tr)['p.value']
    if pchk > alpha:
        ci[0] = 0
    ci[1] = bvec[icu]
    if ci[0] < 0:
        ci[0] = 0
    es = abs(yuenv2(x, y, tr=tr)['Effect.Size'])
    return {'effsize': es, 'alpha': alpha, 'CI': ci}