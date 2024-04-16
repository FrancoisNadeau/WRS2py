def winvar(x, tr=0.2, na_rm=False, STAND=None, *args, **kwargs):
    if na_rm:
        x = [i for i in x if i is not None]
    y = sorted(x)
    n = len(x)
    ibot = int(tr * n) + 1
    itop = len(x) - ibot + 1
    xbot = y[ibot]
    xtop = y[itop]
    y = [xbot if i <= xbot else xtop if i >= xtop else i for i in y]
    winvar = sum([(i - sum(y) / len(y)) ** 2 for i in y]) / len(y)
    return winvar