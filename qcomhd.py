def qcomhd(formula, data, q=[0.1, 0.25, 0.5, 0.75, 0.9], nboot=2000, alpha=0.05, ADJ_CI=True, *args, **kwargs):
    if data is None:
        mf = formula.model.frame()
    else:
        mf = formula.model.frame(data)
    cl = formula.match.call()
    xy = formula.split(model.extract(mf, "response"), mf[,2])
    faclevels = formula.names(xy)
    x = xy[1]
    y = xy[2]
    pv = None
    output = [[None] * 10 for _ in range(len(q))]
    output = pd.DataFrame(output, columns=["q", "n1", "n2", "est.1", "est.2", "est.1_minus_est.2", "ci.low", "ci.up", "p_crit", "p-value"])
    for i in range(len(q)):
        output.loc[i, "q"] = q[i]
        output.loc[i, "n1"] = len(elimna(x))
        output.loc[i, "n2"] = len(elimna(y))
        output.loc[i, "est.1"] = hd(x, q=q[i])
        output.loc[i, "est.2"] = hd(y, q=q[i])
        output.loc[i, "est.1_minus_est.2"] = output.loc[i, "est.1"] - output.loc[i, "est.2"]
        temp = pb2gen1(x, y, nboot=nboot, est=hd, q=q[i], SEED=False, alpha=alpha, pr=False)
        output.loc[i, "ci.low"] = temp["ci"][0]
        output.loc[i, "ci.up"] = temp["ci"][1]
        output.loc[i, "p-value"] = temp["p.value"]
    temp = output["p-value"].argsort(ascending=False)
    zvec = alpha / np.arange(1, len(q)+1)
    output.loc[temp, "p_crit"] = zvec
    if ADJ_CI:
        for i in range(len(q)):
            temp = pb2gen1(x, y, nboot=nboot, est=hd, q=q[i], SEED=False, alpha=output.loc[i, "p_crit"], pr=False)
            output.loc[i, "ci.low"] = temp["ci"][0]
            output.loc[i, "ci.up"] = temp["ci"][1]
            output.loc[i, "p-value"] = temp["p.value"]
    output["signif"] = "YES"
    temp = output["p-value"].argsort(ascending=False)
    for i in range(len(output)):
        if output.loc[temp[i], "p-value"] > output.loc[temp[i], "p_crit"]:
            output.loc[temp[i], "signif"] = "NO"
        if output.loc[temp[i], "p-value"] <= output.loc[temp[i], "p_crit"]:
            break
    output = output.drop(columns=["p_crit"])
    output.columns = ["q", "n1", "n2", "est1", "est2", "est1-est.2", "ci.low", "ci.up", "p.crit", "p.value"]
    result = {"partable": output, "alpha": alpha, "call": cl}
    result = pd.DataFrame(result)
    result["signif"] = "YES"
    temp = result["p.value"].argsort(ascending=False)
    for i in range(len(result)):
        if result.loc[temp[i], "p.value"] > result.loc[temp[i], "alpha"]:
            result.loc[temp[i], "signif"] = "NO"
        if result.loc[temp[i], "p.value"] <= result.loc[temp[i], "alpha"]:
            break
    result = result.drop(columns=["alpha"])
    result.columns = ["q", "n1", "n2", "est1", "est2", "est1-est.2", "ci.low", "ci.up", "p.crit", "p.value"]
    result = result.to_dict()
    result["signif"] = result["signif"].tolist()
    return result