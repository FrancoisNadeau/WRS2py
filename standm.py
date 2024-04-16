import numpy as np

def standm(x, locfun=np.mean, est=np.mean, scat=np.var, *args, **kwargs):
    x = elimna(x)
    x = np.matrix(x)
    m1 = locfun(x, est=est)
    v1 = np.apply_along_axis(scat, 0, x)
    p = x.shape[1]
    for j in range(p):
        x[:, j] = (x[:, j] - m1[j]) / np.sqrt(v1[j])
    return x