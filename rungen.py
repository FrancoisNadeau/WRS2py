import numpy as np

def rungen(x, y, fr=1, est="mom", *args, **kwargs):
    eout = False
    xout = False
    pyhat = True
    est = np.select([est=="mom", est=="onestep", est=="median"], [mom, onestep, median])
    m = np.column_stack((x, y))
    m = elimna(m)
    
    x = m[:, 0]
    y = m[:, 1]
    rmd = np.zeros(len(x))
    for i in range(len(x)):
        rmd[i] = est(y[np.isclose(x, x[i], fr)])
    return rmd