def medpb2(formula, data, nboot=2000, *args):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    xy = split(model.extract(mf, "response"), mf[,2])
    faclevels = names(xy)
    x = xy[[1]]
    y = xy[[2]]
    alpha = 0.05
    x = elimna(x)
    y = elimna(y)
    xx = []
    xx.append(x)
    xx.append(y)
    est1 = median(xx[0])
    est2 = median(xx[1])
    est_dif = median(xx[0]) - median(xx[1])
    crit = alpha / 2
    temp = round(crit * nboot)
    icl = temp + 1
    icu = nboot - temp
    bvec = [[None] * nboot] * 2
    for j in range(2):
        data = matrix(sample(xx[j], size=length(xx[j]) * nboot, replace=True), nrow=nboot)
        bvec[j] = apply(data, 1, median)
    top = bvec[0] - bvec[1]
    test = sum(top < 0) / nboot + 0.5 * sum(top == 0) / nboot
    if test > 0.5:
        test = 1 - test
    top = sort(top)
    ci = [None] * 2
    ci[0] = top[icl]
    ci[1] = top[icu]
    result = {'test': est_dif, 'conf.int': ci, 'p.value': 2 * test, 'call': cl}
    result['class'] = "pb2"
    return result