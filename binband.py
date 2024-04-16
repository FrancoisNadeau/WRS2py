def binband(x, y, KMS=False, alpha=0.05, ADJ_P=False, *args, **kwargs):
    cl = locals()
    x = elimna(x)
    y = elimna(y)
    vals = sorted(set(x).union(y))
    ncon = len(vals)
    n1 = len(x)
    n2 = len(y)
    p_values = []
    adj = 1
    cv = 1
    if not KMS:
        output = [[None] * 6 for _ in range(len(vals))]
        output = pd.DataFrame(output, columns=["Value", "p1.est", "p2.est", "p1-p2", "p.value", "p.crit"])
    if KMS:
        output = [[None] * 8 for _ in range(len(vals))]
        output = pd.DataFrame(output, columns=["Value", "p1.est", "p2.est", "p1-p2", "ci.low", "ci.up", "p.value", "p.crit"])
    for i in range(len(vals)):
        x1 = sum(x == vals[i])
        y1 = sum(y == vals[i])
        if not KMS:
            output.loc[i, "p.value"] = twobinom(x1, n1, y1, n2).p_value
            output.loc[i, "p1.est"] = x1 / n1
            output.loc[i, "p2.est"] = y1 / n2
            output.loc[i, "Value"] = vals[i]
            output.loc[i, "p1-p2"] = output.loc[i, "p1.est"] - output.loc[i, "p2.est"]
        if KMS:
            temp = bi2KMSv2(x1, n1, y1, n2)
            output.loc[i, "Value"] = vals[i]
            output.loc[i, "ci.low"] = temp.ci[0]
            output.loc[i, "ci.up"] = temp.ci[1]
            output.loc[i, "p1.est"] = x1 / n1
            output.loc[i, "p2.est"] = y1 / n2
            output.loc[i, "p1-p2"] = output.loc[i, "p1.est"] - output.loc[i, "p2.est"]
            output.loc[i, "p.value"] = temp.p_value
    ncon = len(vals)
    dvec = [alpha / i for i in range(1, ncon + 1)]
    if ADJ_P:
        mn = max(n1, n2)
        cv = 1
        if ncon != 2:
            if mn > 50:
                cv = 2 - (mn - 50) / 50
                if cv < 1:
                    cv = 1
            if mn <= 50:
                cv = 2
        if KMS:
            flag = output["p.value"] <= 2 * alpha
            output.loc[flag, "p.crit"] = output.loc[flag, "p.crit"] / cv
        if not KMS:
            cv = 1
            flag = output["p.value"] <= 2 * alpha
            if min(n1, n2) < 20 and n1 != n2 and ncon >= 5:
                cv = 2
            output.loc[flag, "p.value"] = output.loc[flag, "p.value"] / cv
    if KMS:
        temp2 = output["p.value"].argsort()[::-1]
        output.loc[temp2, "p.crit"] = dvec
    if not KMS:
        temp2 = output["p.value"].argsort()[::-1]
        output.loc[temp2, "p.crit"] = dvec
    outtable = output.copy()
    outtable.columns = ["Value", "p1.est", "p2.est", "p1-p2"]
    result = {"partable": outtable, "alpha": alpha, "call": cl}
    result = pd.DataFrame(result)
    result.__class__ = "robtab"
    return result