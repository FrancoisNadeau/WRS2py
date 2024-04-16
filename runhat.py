def runhat(x, y, pts=x, est=onestep, fr=1, nmin=1):
    rmd = [None] * len(pts)
    for i in range(len(pts)):
        val = [y[j] for j in range(len(x)) if abs(x[j] - pts[i]) <= fr]
        if len(val) >= nmin:
            rmd[i] = est(val)
    return rmd