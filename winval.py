def winval(x, tr=0.2, ...):
    y = sorted(x)
    n = len(x)
    ibot = int(tr * n) + 1
    itop = len(x) - ibot + 1
    xbot = y[ibot]
    xtop = y[itop]
    winval = [xbot if val <= xbot else val for val in x]
    winval = [xtop if val >= xtop else val for val in winval]
    return winval