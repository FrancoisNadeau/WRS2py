def discstep(formula, data, nboot=500, alpha=0.05, *args, **kwargs):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    x = split(model.extract(mf, "response"), mf[,2])
    vals = [list(set(i)) for i in x]
    vals = sorted(elimna(list2vec(vals)))
    K = len(set(vals))
    n = [len(i) for i in x]
    n = list2vec(n)
    J = len(x)
    if J == 2:
        raise ValueError('For 2 groups use disc2com')
    if J > 5:
        raise ValueError('Designed for 5 groups or less')
    com = com1 = modgen(J)
    lnames = levels(mf[,2])
    com = [lnames[i] for i in com]
    ntest = len(com)
    jp1 = J + 1
    com = com[jp1:len(com)]
    com1 = com1[jp1:len(com1)]
    ntest = len(com)
    mout = pd.DataFrame(np.nan, index=range(ntest), columns=['Groups', 'p-value', 'p.crit'])
    test = []
    for i in range(ntest):
        test.append(discANOVA.sub(x[com[i]]).test)
        nmod = len(com[i]) - 1
        temp = list(range(nmod, -1, -1))
        mout.loc[i, 'Groups'] = '-'.join(com[i])
    mout['p.crit'] = alpha
    xx = []
    pv = np.nan
    jm2 = J - 2
    mout['p.crit'] = alpha
    TB = np.empty((nboot, ntest))
    step1 = discANOVA.sub(x)
    C1 = step1.C1
    HT = []
    for i in range(K):
        HT.append(np.mean(C1[i]))
    for ib in range(nboot):
        xx = []
        for j in range(J):
            temp = rmultinomial(n[j], 1, HT)
            xx.append(np.where(temp[0] == 1)[0])
            for i in range(1, n[j]):
                xx[j].append(np.where(temp[i] == 1)[0])
        for k in range(ntest):
            TB[ib, k] = discANOVA.sub(xx[com1[k]]).test
    for k in range(ntest):
        mout.loc[k, 'p-value'] = 1 - np.mean(test[k] > TB[:, k]) - 0.5 * np.mean(test[k] == TB[:, k])
        pnum = len(com[k])
        pe = 1 - (1 - alpha) ** (pnum / J)
        if len(com[k]) <= jm2:
            mout.loc[k, 'p.crit'] = pe
    outtable = mout.iloc[::-1]
    outtable.reset_index(drop=True, inplace=True)
    result = {'partable': outtable, 'alpha': alpha, 'call': cl}
    result = {'partable': outtable, 'alpha': alpha, 'call': cl}
    result.__class__ = 'robtab'
    return result