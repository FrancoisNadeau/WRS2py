import numpy as np

def apgdis(m, est=np.sum, se=True, *args, **kwargs):
    m = elimna(m)
    temp = 0
    if se:
        for j in range(m.shape[1]):
            m[:, j] = (m[:, j] - np.median(m[:, j])) / np.median(np.abs(m[:, j] - np.median(m[:, j])))
    for j in range(m.shape[1]):
        disx = np.subtract.outer(m[:, j], m[:, j])
        temp += disx ** 2
    temp = np.sqrt(temp)
    dis = np.apply_along_axis(est, 1, temp, *args, **kwargs)
    temp2 = np.argsort(dis)
    center = m[temp2[0], :]
    return {'center': center, 'distance': dis}

