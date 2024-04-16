def med1way(formula, data, iter=1000, *args, **kwargs):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    alpha = 0.05
    crit = None
    SEED = True
    x = split(model.extract(mf, "response"), mf[:, 2])
    grp = range(1, len(x)+1)
    J = len(grp)
    n = [0] * J
    w = [0] * J
    xbar = [0] * J
    for j in range(J):
        xx = [not math.isnan(val) for val in x[j]]
        val = x[j]
        x[j] = [val[i] for i in range(len(val)) if xx[i]]
        w[j] = 1 / msmedse(x[grp[j]], sewarn=False)**2
        xbar[j] = statistics.median(x[grp[j]])
        n[j] = len(x[grp[j]])
    pval = None
    u = sum(w)
    xtil = sum([w[i] * xbar[i] for i in range(J)]) / u
    TEST = sum([w[i] * (xbar[i] - xtil)**2 for i in range(J)]) / (J - 1)
    if math.isnan(crit):
        temp = med1way.crit(n, alpha, SEED=SEED, iter=iter, TEST=TEST)
        crit_val = temp['crit.val']
    if not math.isnan(crit):
        crit_val = crit
    result = {'test': TEST, 'crit.val': crit_val, 'p.value': temp['p.value'], 'call': cl}
    result.__class__ = ['med1way']
    return result