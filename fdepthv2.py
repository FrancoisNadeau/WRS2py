def fdepthv2(m, pts=None, plotit=True):
    m = elimna(m)
    if pts is not None and not np.isnan(pts[0]):
        remm = m
    if not isinstance(m, np.ndarray):
        dep = unidepth(m)
    if isinstance(m, np.ndarray):
        nm = np.shape(m)[0]
        nt = nm
        nm1 = nm + 1
        if pts is not None and not np.isnan(pts[0]):
            if np.shape(m)[1] != np.shape(pts)[1]:
                raise ValueError("Number of columns of m is not equal to number of columns for pts")
            nt = nm + np.shape(pts)[0]
    if np.shape(m)[1] == 1:
        depth = unidepth(m)
    if np.shape(m)[1] > 1:
        m = elimna(m)
        nc = (np.shape(m)[0] ** 2 - np.shape(m)[0]) / 2
        if pts is None:
            mdep = np.zeros((nc, np.shape(m)[0]))
        if pts is not None:
            mdep = np.zeros((nc, np.shape(pts)[0]))
        ic = 0
        for iall in range(1, nm+1):
            for i in range(1, nm+1):
                if iall < i:
                    ic = ic + 1
                    B = m[i-1, :] - m[iall-1, :]
                    dis = np.nan
                    BB = B ** 2
                    bot = np.sum(BB)
                    if bot != 0:
                        if pts is None:
                            for j in range(1, np.shape(m)[0]+1):
                                A = m[j-1, :] - m[iall-1, :]
                                temp = np.sum(A * B) * B / bot
                                dis[j-1] = np.sign(np.sum(A * B)) * np.sqrt(np.sum(temp ** 2))
                        if pts is not None:
                            m = np.vstack((remm, pts))
                            for j in range(1, np.shape(m)[0]+1):
                                A = m[j-1, :] - m[iall-1, :]
                                temp = np.sum(A * B) * B / bot
                                dis[j-1] = np.sign(np.sum(A * B)) * np.sqrt(np.sum(temp ** 2))
                        if pts is None:
                            mdep[ic-1, :] = unidepth(dis)
                        if pts is not None:
                            mdep[ic-1, :] = unidepth(dis[:nm], dis[nm1:np.shape(m)[0]])
                    if bot == 0:
                        mdep[ic-1, :] = np.zeros(np.shape(mdep)[1])
    dep = np.apply_along_axis(np.min, 0, mdep)
    if np.shape(m)[1] == 2 and pts is None:
        flag = chull(m)
        dep[flag] = np.min(dep)
    if np.shape(m)[1] == 2:
        if pts is None and plotit:
            plt.plot(m)
            x = m
            temp = dep
            flag = (temp >= np.median(temp))
            xx = x[flag, :]
            xord = np.argsort(xx[:, 0])
            xx = xx[xord, :]
            temp = chull(xx)
            xord = np.argsort(xx[:, 0])
            xx = xx[xord, :]
            temp = chull(xx)
            plt.lines(xx[temp, :])
            plt.lines(xx[[temp[0], temp[-1]], :])
    return dep