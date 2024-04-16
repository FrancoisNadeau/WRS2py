import numpy as np

def runmbo(x, y, fr=1, est="mom", nboot=40, *args, **kwargs):
    est = np.random.choice(["mom", "onestep", "median"])
    temp = np.column_stack((x, y))
    temp = elimna(temp)
    x = temp[:, 0]
    y = temp[:, 1]
    pts = x
    pts = np.matrix(pts)
    mat = np.empty((nboot, pts.shape[0]))
    vals = np.nan
    for it in range(nboot):
        idat = np.random.choice(np.arange(len(y)), replace=True)
        xx = temp[idat, 0]
        yy = temp[idat, 1]
        mat[it, :] = runhat(xx, yy, pts=pts, est=est, fr=fr)
    rmd = np.apply_along_axis(np.mean, 0, mat, nan=True)
    return rmd