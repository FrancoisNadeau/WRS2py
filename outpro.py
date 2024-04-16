import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import median_absolute_deviation

def outpro(m, gval=np.nan, center=np.nan, plotit=True, op=True, MM=False, cop=3,
           xlab="VAR 1", ylab="VAR 2", STAND=True, tr=0.2, q=0.5, pr=True, **kwargs):
    """
    Detect outliers using a modified Stahel-Donoho projection method.

    Determine center of data cloud, for each point,
    connect it with center, project points onto this line
    and use distances between projected points to detect
    outliers. A boxplot method is used on the
    projected distances.
    #
    plotit=TRUE creates a scatterplot when working with
    bivariate data.
    #
    op=T
    means the .5 depth contour is plotted
    based on data with outliers removed.
    #
    op=F
    means .5 depth contour is plotted without removing outliers.
    #
     MM=F  Use interquatile range when checking for outliers
     MM=T  uses MAD.
    #
     If value for center is not specified,
     there are four options for computing the center of the
     cloud of points when computing projections:
    #
     cop=2 uses MCD center
     cop=3 uses median of the marginal distributions.
     cop=4 uses MVE center
     cop=5 uses TBS
     cop=6 uses rmba (Olive's median ball algorithm) cop=7 uses the spatial (L1) median
    #
     args q and tr having are not used by this function. They are included to deal
     with situations where smoothers have optional arguments for q and tr
    #
     When using cop=2, 3 or 4, default critical value for outliers
     is square root of the .975 quantile of a
     chi-squared distribution with p degrees
     of freedom.
    #
     STAND=T means that marginal distributions are standardized before
     checking for outliers.
    #
     Donoho-Gasko (Tukey) median is marked with a cross, +.
    """

    m = np.array(m)
    if pr:
        if not STAND:
            if m.shape[1] > 1:
                print("STAND=FALSE. If measures are on different scales, might want to use STAND=TRUE")
    m = elimna(m)
    m = np.array(m)
    nv = m.shape[0]
    if m.shape[1] == 1:
        dis = (m - np.median(m, axis=0, nan_policy='omit'))**2 / median_absolute_deviation(m, axis=0, nan_policy='omit')**2
        dis = np.sqrt(dis)
        dis[np.isnan(dis)] = 0
        crit = np.sqrt(chi2.ppf(0.975, 1))
        chk = np.where(dis > crit, 1, 0)
        vec = np.arange(1, m.shape[0]+1)
        outid = vec[chk == 1]
        keep = vec[chk == 0]
    if m.shape[1] > 1:
        if STAND:
            m = standm(m, est=np.median, scat=median_absolute_deviation)
        if np.isnan(gval) and cop == 1:
            gval = np.sqrt(chi2.ppf(0.95, m.shape[1]))
        if np.isnan(gval) and cop != 1:
            gval = np.sqrt(chi2.ppf(0.975, m.shape[1]))
        if cop == 1 and np.isnan(center[0]):
            if m.shape[1] > 2:
                center = dmean(m, tr=0.5, cop=1)
            if m.shape[1] == 2:
                tempd = np.empty(m.shape[0])
                for i in range(m.shape[0]):
                    tempd[i] = depth(m[i, 0], m[i, 1], m)
                mdep = np.max(tempd)
                flag = (tempd == mdep)
                if np.sum(flag) == 1:
                    center = m[flag, :]
                if np.sum(flag) > 1:
                    center = np.mean(m[flag, :], axis=0)
        if cop == 2 and np.isnan(center[0]):
            center = cov.mcd(m).center
        if cop == 4 and np.isnan(center[0]):
            center = cov.mve(m).center
        if cop == 3 and np.isnan(center[0]):
            center = np.apply_along_axis(np.median, 0, m)
        if cop == 6 and np.isnan(center[0]):
            center = rmba(m).center
        if cop == 7 and np.isnan(center[0]):
            center = spat(m)
        flag = np.zeros(m.shape[0])
        outid = np.nan
        vec = np.arange(1, m.shape[0]+1)
        for i in range(m.shape[0]):
            B = m[i, :] - center
            dis = np.empty(m.shape[0])
            BB = B**2
            bot = np.sum(BB)
            if bot != 0:
                for j in range(m.shape[0]):
                    A = m[j, :] - center
                    temp = np.sum(A*B)*B/bot
                    dis[j] = np.sqrt(np.sum(temp**2))
                temp = idealf(dis)
                if not MM:
                    cu = np.median(dis) + gval*(temp['qu'] - temp['ql'])
                if MM:
                    cu = np.median(dis) + gval*median_absolute_deviation(dis)
                outid = np.nan
                temp2 = (dis > cu)
                flag[temp2] = 1
        if np.sum(flag) == 0:
            outid = np.nan
        if np.sum(flag) > 0:
            flag = (flag == 1)
            outid = vec[flag]
        idv = np.arange(1, m.shape[0]+1)
        keep = idv[~flag]
        if m.shape[1] == 2:
            if plotit:
                plt.plot(m[:, 0], m[:, 1], 'n', label=xlab, ylabel=ylab)
                plt.scatter(m[keep, 0], m[keep, 1], marker='*')
                if len(outid) > 0:
                    plt.scatter(m[outid, 0], m[outid, 1], marker='o')
                if op:
                    tempd = np.empty(m[keep, :].shape[0])
                    mm = m[keep, :]
                    for i in range(mm.shape[0]):
                        tempd[i] = depth(mm[i, 0], mm[i, 1], mm)
                    mdep = np.max(tempd)
                    flag = (tempd == mdep)
                    if np.sum(flag) == 1:
                        center = mm[flag, :]
                    if np.sum(flag) > 1:
                        center = np.mean(mm[flag, :], axis=0)
                    m = mm
                plt.scatter(center[0], center[1], marker='+')
                x = m
                temp = fdepth(m, plotit=False)
                flag = (temp >= np.median(temp))
                xx = x[flag, :]
                xord = np.argsort(xx[:, 0])
                xx = xx[xord, :]
                temp = chull(xx)
                plt.plot(xx[temp, :])
                plt.plot(xx[[temp[0], temp[-1]], :])
    return {'n': nv, 'n.out': len(outid), 'out.id': outid, 'keep': keep}