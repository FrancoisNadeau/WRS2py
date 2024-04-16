def outbox(x, mbox=False, gval=float('nan'), plotit=False, STAND=False):
    x = [i for i in x if not math.isnan(i)]
    if plotit:
        plt.boxplot(x)
    n = len(x)
    temp = idealf(x)
    if mbox:
        if math.isnan(gval):
            gval = (17.63 * n - 23.64) / (7.74 * n - 3.71)
        cl = np.median(x) - gval * (temp['qu'] - temp['ql'])
        cu = np.median(x) + gval * (temp['qu'] - temp['ql'])
    if not mbox:
        if math.isnan(gval):
            gval = 1.5
        cl = temp['ql'] - gval * (temp['qu'] - temp['ql'])
        cu = temp['qu'] + gval * (temp['qu'] - temp['ql'])
    flag = []
    outid = []
    vec = list(range(1, n+1))
    for i in range(n):
        flag.append(x[i] < cl or x[i] > cu)
    if sum(flag) == 0:
        outid = None
    if sum(flag) > 0:
        outid = [vec[i] for i in range(n) if flag[i]]
    keep = [vec[i] for i in range(n) if not flag[i]]
    outval = [x[i] for i in range(n) if flag[i]]
    n_out = len(outid)
    return {'out.val': outval, 'out.id': outid, 'keep': keep, 'n': n, 'n.out': n_out, 'cl': cl, 'cu': cu}