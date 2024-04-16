def Qanova(formula, data, q=0.5, nboot=600, *args, **kwargs):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    x = split(model.extract(mf, "response"), mf[,2])
    op = 3
    MC = False
    chkcar = []
    for j in range(len(x)):
        chkcar.append(len(set(x[j])))
    nc = len(pbadepth1(x, est=hd, q=q[0], allp=True, SEED=False, op=op, nboot=nboot, MC=MC, na.rm=True)['psihat'])
    psimat = [[None] * nc for _ in range(len(q))]
    pvals = [None] * len(q)
    for i in range(len(q)):
        output = pbadepth1(x, est=hd, q=q[i], allp=True, SEED=False, op=op, nboot=nboot, MC=MC, na.rm=True)
        psimat[i] = output['psihat']
        pvals[i] = output['p.value']
    psidf = pd.DataFrame(psimat)
    psidf.index = ["q = " + str(q_val) for q_val in q]
    psidf.columns = ["con" + str(i) for i in range(1, nc+1)]
    pvalues = pd.DataFrame({'p-value': pvals, 'p-adj': p.adjust(pvals, method='hochberg')})
    pvalues.index = ["q = " + str(q_val) for q_val in q]
    pvalues.columns = ["p-value", "p-adj"]
    result = {'psihat': psidf, 'p.value': pvalues, 'contrasts': output['con'], 'call': cl}
    result.__class__ = "qanova"
    return result