import numpy as np
from scipy.spatial import ConvexHull

def fdepth(m, pts=None, plotit=True, cop=3, center=None, xlab="VAR 1", ylab="VAR 2"):
    if cop != 2 and cop != 3 and cop != 4:
        raise ValueError("Only cop=2, 3 or 4 is allowed")
    if isinstance(m, list):
        raise ValueError("Store data in a matrix; might use function listm")
    m = np.array(m)
    if pts is not None:
        pts = np.array(pts)
        if pts.shape[1] != m.shape[1]:
            raise ValueError("Number of columns of m is not equal to number of columns for pts")
    m = elimna(m)
    m = np.array(m)
    if m.shape[1] == 1:
        dep = unidepth(np.vectorize(m[:, 0]), pts=pts)
    if m.shape[1] > 1:
        if center is None:
            if cop == 2:
                center = cov.mcd(m).center
            if cop == 4:
                center = cov.mve(m).center
            if cop == 3:
                center = np.median(m, axis=0)
        if pts is None:
            mdep = np.empty((m.shape[0], m.shape[0]))
        if pts is not None:
            mdep = np.empty((m.shape[0], pts.shape[0]))
        for i in range(m.shape[0]):
            B = m[i, :] - center
            dis = np.empty(m.shape[0])
            BB = B ** 2
            bot = np.sum(BB)
            if bot != 0:
                if pts is None:
                    for j in range(m.shape[0]):
                        A = m[j, :] - center
                        temp = np.sum(A * B) * B / bot
                        dis[j] = np.sign(np.sum(A * B)) * np.sqrt(np.sum(temp ** 2))
                if pts is not None:
                    m = np.vstack((remm, pts))
                    for j in range(m.shape[0]):
                        A = m[j, :] - center
                        temp = np.sum(A * B) * B / bot
                        dis[j] = np.sign(np.sum(A * B)) * np.sqrt(np.sum(temp ** 2))
                if pts is None:
                    mdep[i, :] = unidepth(dis)
                if pts is not None:
                    mdep[i, :] = unidepth(dis[:m.shape[0]], dis[m.shape[0]:])
            if bot == 0:
                mdep[i, :] = np.zeros(mdep.shape[1])
        dep = np.min(mdep, axis=1)
        if m.shape[1] == 2 and pts is None:
            flag = ConvexHull(m).vertices
            dep[flag] = np.min(dep)
    if m.shape[1] == 2:
        if pts is None and plotit:
            plt.plot(m[:, 0], m[:, 1])
            plt.scatter(center[0], center[1], marker="+")
            x = m
            temp = dep
            flag = temp >= np.median(temp)
            xx = x[flag, :]
            xord = np.argsort(xx[:, 0])
            xx = xx[xord, :]
            temp = ConvexHull(xx).vertices
            xord = np.argsort(xx[:, 0])
            xx = xx[xord, :]
            temp = ConvexHull(xx).vertices
            plt.plot(xx[temp, :])
            plt.plot(xx[[temp[0], temp[-1]], :])
    dep = np.round(dep * m.shape[0]) / m.shape[0]
    return dep