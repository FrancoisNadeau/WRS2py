def discANOVA(formula, data, nboot=500, *args, **kwargs):
    if 'data' not in kwargs:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    x = split(model.extract(mf, "response"), mf[,2])
    
    vals = lapply(x, unique)
    vals = sort(elimna(list2vec(vals)))
    K = len(unique(vals))
    n = lapply(x, length)
    n = list2vec(n)
    J = length(x)
    step1 = discANOVA.sub(x)
    test = step1$test
    C1 = step1$C1
    HT = NULL
    for i in range(1, K):
        HT[i] = mean(C1[i,])
    tv = NULL
    TB = NA
    VP = NA
    B1hat = NA
    xx = list()
    for ib in range(1, nboot):
        xx = list()
        for j in range(1, J):
            temp = rmultinomial(n[j], 1, HT)
            xx[[j]] = which(temp[1,] == 1)
            for i in range(2, n[j]):
                xx[[j]][i] = which(temp[i,] == 1)
        TB[ib] = discANOVA.sub(xx)$test
    pv = 1 - mean(test > TB) - .5 * mean(test == TB)
    result = {'test': test, 'crit.val': NA, 'p.value': pv, 'call': cl}
    result['class'] = "med1way"
    return result