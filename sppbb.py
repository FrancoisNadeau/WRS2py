def sppbb(formula, id, data, est="mom", nboot=500, *args, **kwargs):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    est = match.arg(est, c("mom", "onestep", "median"), several.ok=False)
    mf1 = match.call()
    m = match(c("formula", "data", "id"), names(mf1), 0L)
    mf1 = mf1[c(1L, m)]
    mf1.drop.unused.levels = True
    mf1[1L] = quote(stats.model.frame)
    mf1 = eval(mf1, parent.frame())
    random1 = mf1[, "(id)"]
    depvar = colnames(mf)[0]
    if all(length(table(random1)) == table(mf[, 2])):
        ranvar = colnames(mf)[2]
        fixvar = colnames(mf)[1]
    else:
        ranvar = colnames(mf)[1]
        fixvar = colnames(mf)[2]
    MC = False
    K = length(table(mf[, ranvar]))
    J = length(table(mf[, fixvar]))
    p = J * K
    grp = range(p)
    est = get(est)
    fixsplit = split(mf[, depvar], mf[, fixvar])
    indsplit = split(mf[, ranvar], mf[, fixvar])
    dattemp = mapply(split, fixsplit, indsplit, SIMPLIFY=False)
    data = do.call(c, dattemp)
    x = data
    jp = 1 - K
    kv = 0
    kv2 = 0
    for j in range(J):
        jp = jp + K
        xmat = matrix(NA, ncol=K, nrow=len(x[jp]))
        for k in range(K):
            kv = kv + 1
            xmat[:, k] = x[kv]
        xmat = elimna(xmat)
        for k in range(K):
            kv2 = kv2 + 1
            x[kv2] = xmat[:, k]
    xx = x
    nvec = NA
    jp = 1 - K
    for j in range(J):
        jp = jp + K
        nvec[j] = len(x[jp])
    x = matrix(NA, nrow=nvec[0], ncol=K)
    for k in range(K):
        x[:, k] = xx[k]
    kc = K
    for j in range(1, J):
        temp = matrix(NA, nrow=nvec[j], ncol=K)
        for k in range(K):
            kc = kc + 1
            temp[:, k] = xx[kc]
        x = rbind(x, temp)
    temp = rmdzero(x, est=est, nboot=nboot)
    tvec1 = data.frame(Estimate=temp.center)
    tnames = apply(combn(levels(mf[, ranvar]), 2), 2, paste0, collapse="-")
    rownames(tvec1) = tnames
    result = list(test=tvec1, p.value=temp.p.value, call=cl)
    class(result) = c("spp")
    result