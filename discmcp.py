def discmcp(formula, data, alpha = 0.05, nboot = 500, *args):
    if data is None:
        mf = model.frame(formula)
    else:
        mf = model.frame(formula, data)
    cl = match.call()
    x = split(model.extract(mf, "response"), mf[,2])
    J = len(x)
    ncon = (J**2 - J) / 2
    Jm = J - 1
    
    dvec = [alpha / i for i in range(1, ncon+1)]
    output = pd.DataFrame(columns=['Group 1', 'Group 2', 'p.value', 'p.crit'])
    ic = 0
    for j in range(1, J+1):
        for k in range(1, J+1):
            if j < k:
                ic += 1
                output.loc[ic, 'Group 1'] = levels(mf[,2])[j]
                output.loc[ic, 'Group 2'] = levels(mf[,2])[k]
                output.loc[ic, 'p.value'] = disc2com(x[j], x[k], simulate.p.value = True, B=nboot)['p.value']
    
    temp2 = output['p.value'].argsort(ascending=False)
    zvec = dvec[:ncon]
    output.loc[temp2, 'p.crit'] = zvec
    num_sig = sum(output['p.value'] <= output['p.crit'])
    outtable = output
    result = {'partable': outtable, 'alpha': alpha, 'call': cl}
    result.__class__ = "robtab"
    return result