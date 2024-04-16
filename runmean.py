import numpy as np

def runmean(x, y, fr=1, tr=0.2, **kwargs):
    eout = False
    xout = False
    pyhat = True
    
    temp = np.column_stack((x, y))
    temp = elimna(temp)
    
    x = temp[:, 0]
    y = temp[:, 1]
    
    rmd = np.zeros(len(x))
    for i in range(len(x)):
        rmd[i] = np.mean(y[np.isclose(x, x[i], atol=fr)], tr)
    
    return rmd