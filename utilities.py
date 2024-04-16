import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
# from scipy.spatial.
from scipy.stats import norm, Covariance as cov
from scipy.stats import median_abs_deviation as mad
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, t

from dmean import dmean
from elimna import elimna
from onestep import onestep
# from mad import mad
from mest import mest
from mom import mom
from dnormvar import dnormvar
from smean import smean
from tmean import tmean
from wincor import wincor
from winvar import winvar
# from mahalanobis import mahalanobis


def matl(x):
    J = len(x)
    nval = np.zeros(J)
    for j in range(J):
        nval[j] = len(x[j])
    temp = np.full((max(nval), J), np.nan)
    for j in range(J):
        temp[:int(nval[j]), j] = x[j]
    return temp

list2mat = matl

def list2vec(x):
    if not isinstance(x, list):
        raise ValueError("x should have list mode")
    res = np.asarray(matl(x)).flatten()
    return res

def discANOVA_sub(x):
    x = [elimna(i) for i in x]
    vals = [np.unique(i) for i in x]
    vals = np.sort(elimna(list2vec(vals)))
    n = [len(i) for i in x]
    n = list2vec(n)
    K = len(vals)
    J = len(x)
    C1 = np.zeros((K, J))
    for j in range(J):
        for i in range(K):
            C1[i, j] = C1[i, j] + np.sum(x[j] == vals[i])
        C1[:, j] = C1[:, j] / n[j]
    test = 0
    for i in range(K):
        test = test + np.var(C1[i, :])
    return {"test": test, "C1": C1}

def modgen(p, adz=False):
    model = {}
    if p > 5:
        raise ValueError("Current version is limited to 5 predictors")
    if p == 1:
        model[1] = 1
    if p == 2:
        model[1] = 1
        model[2] = 2
        model[3] = [1, 2]
    if p == 3:
        for i in range(1, 4):
            model[i] = i
        model[4] = [1, 2]
        model[5] = [1, 3]
        model[6] = [2, 3]
        model[7] = [1, 2, 3]
    if p == 4:
        for i in range(1, 5):
            model[i] = i
        model[5] = [1, 2]
        model[6] = [1, 3]
        model[7] = [1, 4]
        model[8] = [2, 3]
        model[9] = [2, 4]
        model[10] = [3, 4]
        model[11] = [1, 2, 3]
        model[12] = [1, 2, 4]
        model[13] = [1, 3, 4]
        model[14] = [2, 3, 4]
        model[15] = [1, 2, 3, 4]
    if p == 5:
        for i in range(1, 6):
            model[i] = i
        model[6] = [1, 2]
        model[7] = [1, 3]
        model[8] = [1, 4]
        model[9] = [1, 5]
        model[10] = [2, 3]
        model[11] = [2, 4]
        model[12] = [2, 5]
        model[13] = [3, 4]
        model[14] = [3, 5]
        model[15] = [4, 5]
        model[16] = [1, 2, 3]
        model[17] = [1, 2, 4]
        model[18] = [1, 2, 5]
        model[19] = [1, 3, 4]
        model[20] = [1, 3, 5]
        model[21] = [1, 4, 5]
        model[22] = [2, 3, 4]
        model[23] = [2, 3, 5]
        model[24] = [2, 4, 5]
        model[25] = [3, 4, 5]
        model[26] = [1, 2, 3, 4]
        model[27] = [1, 2, 3, 5]
        model[28] = [1, 2, 4, 5]
        model[29] = [1, 3, 4, 5]
        model[30] = [2, 3, 4, 5]
        model[31] = [1, 2, 3, 4, 5]
    if adz:
        ic = len(model) + 1
        model[ic] = 0
    return model

def pb2gen1(x, y, alpha=0.05, nboot=2000, est=onestep, SEED=True, pr=False, **kwargs):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if SEED:
        np.random.seed(2)
    datax = np.random.choice(x, size=len(x) * nboot, replace=True).reshape(nboot, -1)
    datay = np.random.choice(y, size=len(y) * nboot, replace=True).reshape(nboot, -1)
    bvecx = np.apply_along_axis(est, 1, datax, **kwargs)
    bvecy = np.apply_along_axis(est, 1, datay, **kwargs)
    bvec = np.sort(bvecx - bvecy)
    low = int((alpha / 2) * nboot) + 1
    up = nboot - low
    temp = (np.sum(bvec < 0) / nboot) + (np.sum(bvec == 0) / (2 * nboot))
    sig_level = 2 * (min(temp, 1 - temp))
    se = np.var(bvec)
    return {"est.1": est(x, **kwargs), "est.2": est(y, **kwargs), "est.dif": est(x, **kwargs) - est(y, **kwargs), "ci": [bvec[low], bvec[up]], "p.value": sig_level, "sq.se": se, "n1": len(x), "n2": len(y)}

def bootdpci(x, y, est=onestep, nboot=np.nan, alpha=0.05, plotit=True, dif=True, BA=False, SR=True, **kwargs):
    okay = False
    if est == onestep:
        okay = True
    if est == mom:
        okay = True
    if not okay:
        SR = False
    output = rmmcppb(x, y, est=est, nboot=nboot, alpha=alpha, SR=SR, plotit=plotit, dif=dif, BA=BA, **kwargs)["output"]
    return {"output": output}

def rmmcppb(x, y=None, alpha=0.05, con=0, est=onestep, plotit=True, dif=True, grp=np.nan, nboot=np.nan, BA=False, hoch=False, xlab="Group 1", ylab="Group 2", pr=True, SEED=True, SR=False, **kwargs):
    if hoch:
        SR = False
    if SR:
        okay = False
        if est == onestep:
            okay = True
        if est == mom:
            okay = True
        SR = okay
    if dif:
        if pr:
            print("dif=T, so analysis is done on difference scores")
        temp = rmmcppbd(x, y=y, alpha=0.05, con=con, est=est, plotit=plotit, grp=grp, nboot=nboot, hoch=True, **kwargs)
        output = temp["output"]
        con = temp["con"]
    if not dif:
        if pr:
            print("dif=F, so analysis is done on marginal distributions")
            if not BA:
                if est == onestep:
                    print("With M-estimator or MOM, suggest using BA=T and hoch=T")
                if est == mom:
                    print("With M-estimator or MOM, suggest using BA=T and hoch=T")
        if y is not None:
            x = np.column_stack((x, y))
        if not isinstance(x, list) and not isinstance(x, np.ndarray):
            raise ValueError("Data must be stored in a matrix or in list mode.")
        if isinstance(x, list):
            if isinstance(con, np.ndarray):
                if len(x) != con.shape[0]:
                    raise ValueError("The number of rows in con is not equal to the number of groups.")
        if isinstance(x, list):
            mat = matl(x)
        if isinstance(x, np.ndarray) and isinstance(con, np.ndarray):
            if x.shape[1] != con.shape[0]:
                raise ValueError("The number of rows in con is not equal to the number of groups.")
            mat = x
        if isinstance(x, np.ndarray):
            mat = x
        if not np.isnan(grp).any():
            mat = mat[:, grp]
        mat = elimna(mat)
        x = mat
        J = mat.shape[1]
        xcen = x.copy()
        for j in range(J):
            xcen[:, j] = x[:, j] - est(x[:, j], **kwargs)
        Jm = J - 1
        if np.sum(con ** 2) == 0:
            d = (J ** 2 - J) / 2
            con = np.zeros((J, int(d)))
            id = 0
            for j in range(Jm):
                jp = j + 1
                for k in range(jp, J):
                    id = id + 1
                    con[j, id-1] = 1
                    con[k, id-1] = -1
        d = con.shape[1]
        if np.isnan(nboot):
            if d <= 4:
                nboot = 1000
            if d > 4:
                nboot = 5000
        n = mat.shape[0]
        crit_vec = alpha / np.arange(1, d+1)
        connum = con.shape[1]
        if SEED:
            np.random.seed(2)
        xbars = np.apply_along_axis(est, 0, mat, **kwargs)
        psidat = np.zeros(connum)
        for ic in range(connum):
            psidat[ic] = np.sum(con[:, ic] * xbars)
        psihat = np.zeros((connum, nboot))
        psihatcen = np.zeros((connum, nboot))
        bvec = np.zeros((nboot, J))
        bveccen = np.zeros((nboot, J))
        if pr:
            print("Taking bootstrap samples. Please wait.")
        data = np.random.choice(n, size=n * nboot, replace=True).reshape(nboot, -1)
        for ib in range(nboot):
            bvec[ib, :] = np.apply_along_axis(est, 1, x[data[ib, :], :], **kwargs)
            bveccen[ib, :] = np.apply_along_axis(est, 1, xcen[data[ib, :], :], **kwargs)
        test = np.ones(connum)
        bias = np.zeros(connum)
        for ic in range(connum):
            psihat[ic, :] = np.apply_along_axis(bptdpsi, 1, bvec, con[:, ic])
            psihatcen[ic, :] = np.apply_along_axis(bptdpsi, 1, bveccen, con[:, ic])
            bias[ic] = np.sum(psihatcen[ic, :] > 0) / nboot - 0.5
            ptemp = (np.sum(psihat[ic, :] > 0) + 0.5 * np.sum(psihat[ic, :] == 0)) / nboot
            if BA:
                test[ic] = ptemp - 0.1 * bias[ic]
            if not BA:
                test[ic] = ptemp
            test[ic] = min(test[ic], 1 - test[ic])
            test[ic] = max(test[ic], 0)
        test = 2 * test
        ncon = con.shape[1]
        dvec = alpha / np.arange(1, ncon+1)
        if SR:
            if alpha == 0.05:
                dvec = np.array([0.025, 0.025, 0.0169, 0.0127, 0.0102, 0.00851, 0.0073, 0.00639, 0.00568, 0.00511])
                dvecba = np.array([0.05, 0.025, 0.0169, 0.0127, 0.0102, 0.00851, 0.0073, 0.00639, 0.00568, 0.00511])
                if ncon > 10:
                    avec = 0.05 / np.arange(11, ncon+1)
                    dvec = np.concatenate((dvec, avec))
            if alpha == 0.01:
                dvec = np.array([0.005, 0.005, 0.00334, 0.00251, 0.00201, 0.00167, 0.00143, 0.00126, 0.00112, 0.00101])
                dvecba = np.array([0.01, 0.005, 0.00334, 0.00251, 0.00201, 0.00167, 0.00143, 0.00126, 0.00112, 0.00101])
                if ncon > 10:
                    avec = 0.01 / np.arange(11, ncon+1)
                    dvec = np.concatenate((dvec, avec))
            if alpha != 0.05 and alpha != 0.01:
                dvec = alpha / np.arange(1, ncon+1)
                dvecba = dvec
                dvec[1] = alpha
        if hoch:
            dvec = alpha / np.arange(1, ncon+1)
        dvecba = dvec
        if plotit and bvec.shape[1] == 2:
            z = np.zeros(2)
            one = np.ones(2)
            plt.plot(np.vstack((bvec, z, one)).T)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.show()
            totv = np.apply_along_axis(est, 0, x, **kwargs)
            cmat = np.cov(bvec.T)
            dis = mahalanobis(bvec, totv, cmat)
            temp_dis = np.argsort(dis)
            ic = int((1 - alpha) * nboot)
            xx = bvec[temp_dis[:ic], :]
            xord = np.argsort(xx[:, 0])
            xx = xx[xord, :]
            temp = np.convex_hull(xx)
            plt.plot(xx[temp, :])
            plt.plot(xx[[temp[0], temp[-1]], :])
            plt.plot([0, 1], [0, 1], 'k-')
            plt.show()
        temp2 = np.argsort(0 - test)
        ncon = con.shape[1]
        zvec = dvec[:ncon]
        if BA:
            zvec = dvecba[:ncon]
        sigvec = (test[temp2] >= zvec)
        output = np.zeros((connum, 6))
        output[:, 0] = np.arange(1, connum+1)
        output[:, 1] = np.sum(con * xbars, axis=0)
        output[:, 2] = test
        temp = np.sort(psihat, axis=1)
        icl = int(alpha * nboot / 2) + 1
        icu = nboot - (icl - 1)
        output[:, 3] = zvec
        output[:, 4] = temp[:, icl-1]
        output[:, 5] = temp[:, icu-1]
    num_sig = np.sum(output[:, 2] <= output[:, 3])
    return {"output": output, "con": con, "num_sig": num_sig}

def bptdpsi(x, con):
    return np.sum(con * x)

def rmmismcp(x, y=np.nan, alpha=0.05, con=0, est=tmean, plotit=True, grp=np.nan, nboot=500, SEED=True, xlab="Group 1", ylab="Group 2", pr=False, **kwargs):
    if not np.isnan(y[0]):
        x = np.column_stack((x, y))
    if isinstance(x, list):
        x = matl(x)
    if not isinstance(x, list) and not isinstance(x, np.ndarray):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    if isinstance(x, list):
        if isinstance(con, np.ndarray):
            if len(x) != con.shape[0]:
                raise ValueError("The number of rows in con is not equal to the number of groups.")
    if isinstance(x, list):
        mat = matl(x)
    if isinstance(x, np.ndarray) and isinstance(con, np.ndarray):
        if x.shape[1] != con.shape[0]:
            raise ValueError("The number of rows in con is not equal to the number of groups.")
        mat = x
    J = mat.shape[1]
    Jm = J - 1
    flag_con = False
    if np.sum(con ** 2) == 0:
        flag_con = True
        d = (J ** 2 - J) / 2
        con = np.zeros((J, int(d)))
        id = 0
        for j in range(Jm):
            jp = j + 1
            for k in range(jp, J):
                id = id + 1
                con[j, id-1] = 1
                con[k, id-1] = -1
    d = con.shape[1]
    n = mat.shape[0]
    crit_vec = alpha / np.arange(1, d+1)
    connum = con.shape[1]
    if SEED:
        np.random.seed(2)
    xbars = np.apply_along_axis(est, 0, mat, na_rm=True, **kwargs)
    psidat = np.zeros(connum)
    bveccen = np.zeros((nboot, J))
    for ic in range(connum):
        psidat[ic] = np.sum(con[:, ic] * xbars)
    psihat = np.zeros((connum, nboot))
    psihatcen = np.zeros((connum, nboot))
    bvec = np.zeros((nboot, J))
    if pr:
        print("Taking bootstrap samples. Please wait.")
    data = np.random.choice(n, size=n * nboot, replace=True).reshape(nboot, -1)
    for ib in range(nboot):
        bvec[ib, :] = np.apply_along_axis(est, 1, mat[data[ib, :], :], na_rm=True, **kwargs)
    test = np.ones(connum)
    for ic in range(connum):
        for ib in range(nboot):
            psihat[ic, ib] = np.sum(con[:, ic] * bvec[ib, :])
        matcon = np.concatenate(([0], psihat[ic, :]))
        dis = np.mean((psihat[ic, :] < 0)) + 0.5 * np.mean((psihat[ic, :] == 0))
        test[ic] = 2 * min(dis, 1 - dis)
    ncon = con.shape[1]
    dvec = alpha / np.arange(1, ncon+1)
    if plotit and bvec.shape[1] == 2:
        z = np.zeros(2)
        one = np.ones(2)
        plt.plot(np.vstack((bvec, z, one)).T)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
        totv = np.apply_along_axis(est, 0, mat, na_rm=True, **kwargs)
        cmat = np.cov(bvec.T)
        dis = mahalanobis(bvec, totv, cmat)
        temp_dis = np.argsort(dis)
        ic = int((1 - alpha) * nboot)
        xx = bvec[temp_dis[:ic], :]
        xord = np.argsort(xx[:, 0])
        xx = xx[xord, :]
        temp = np.convex_hull(xx)
        plt.plot(xx[temp, :])
        plt.plot(xx[[temp[0], temp[-1]], :])
        plt.plot([0, 1], [0, 1], 'k-')
        plt.show()
    temp2 = np.argsort(0 - test)
    ncon = con.shape[1]
    zvec = dvec[:ncon]
    sigvec = (test[temp2] >= zvec)
    output = np.zeros((connum, 6))
    output[:, 0] = np.arange(1, connum+1)
    output[:, 1] = np.sum(con * xbars, axis=0)
    output[:, 2] = test
    output[:, 3] = zvec
    temp = np.sort(psihat, axis=1)
    icl = int(output[0, 3] * nboot / 2) + 1
    icu = nboot - (icl - 1)
    output[:, 4] = temp[:, icl-1]
    output[:, 5] = temp[:, icu-1]
    if not flag_con:
        pass
    if flag_con:
        CC = (J ** 2 - J) / 2
        test = np.zeros((CC, 7))
        test[:, 0] = np.repeat(np.arange(1, J+1), J-1)
        test[:, 1] = np.tile(np.arange(2, J+1), J-1)
        test[:, 2:5] = output[:, 1:4]
        test[:, 6:8] = output[:, 4:6]
        con = None
    if not flag_con:
        test = output
    if flag_con:
        num_sig = np.sum(test[:, 3] <= test[:, 4])
    if not flag_con:
        num_sig = np.sum(test[:, 2] <= test[:, 3])
    return {"output": test, "con": con, "num_sig": num_sig}


def rmmcppbd(x, y=None, alpha=0.05, con=0, est=np.mean, plotit=True, grp=np.nan, nboot=np.nan,
             hoch=True, SEED=True, *args):
    
    if y is not None:
        x = np.column_stack((x, y))
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    if isinstance(x, list):
        if isinstance(con, np.ndarray):
            if len(x) != con.shape[0]:
                raise ValueError("The number of rows in con is not equal to the number of groups.")
        x = np.vstack(x)
    if isinstance(x, np.ndarray) and isinstance(con, np.ndarray):
        if x.shape[1] != con.shape[0]:
            raise ValueError("The number of rows in con is not equal to the number of groups.")
    if not np.isnan(np.sum(grp)):
        x = x[:, grp]
    mat = x
    J = mat.shape[1]
    n = mat.shape[0]
    if n >= 80:
        hoch = True
    Jm = J - 1
    if np.sum(con ** 2) == 0:
        d = (J ** 2 - J) / 2
        con = np.zeros((J, int(d)))
        id = 0
        for j in range(Jm):
            jp = j + 1
            for k in range(jp, J):
                con[j, id] = 1
                con[k, id] = -1
                id += 1
    d = con.shape[1]
    if np.isnan(nboot):
        nboot = 5000
        if d <= 10:
            nboot = 3000
        if d <= 6:
            nboot = 2000
        if d <= 4:
            nboot = 1000
    crit_vec = alpha / np.arange(1, d + 1)
    connum = con.shape[1]
    
    xx = np.dot(x, con)
    if SEED:
        np.random.seed(2)
    
    psihat = np.zeros((connum, nboot))
    for ib in range(nboot):
        sample_indices = np.random.choice(n, size=n, replace=True)
        if xx.shape[1] == 1:
            psihat[0, ib] = est(xx[sample_indices, :], *args)
        else:
            for col in range(xx.shape[1]):
                psihat[col, ib] = est(xx[sample_indices, col], *args)
    
    test = np.ones(connum)
    for ic in range(connum):
        test[ic] = (np.sum(psihat[ic, :] > 0) + 0.5 * np.sum(psihat[ic, :] == 0)) / nboot
        test[ic] = min(test[ic], 1 - test[ic])
    test = 2 * test
    
    if alpha == 0.05:
        dvec = np.array([0.025, 0.025, 0.0169, 0.0127, 0.0102, 0.00851, 0.0073, 0.00639, 0.00568, 0.00511])
        if connum > 10:
            avec = 0.05 / np.arange(11, connum + 1)
            dvec = np.concatenate((dvec, avec))
    elif alpha == 0.01:
        dvec = np.array([0.005, 0.005, 0.00334, 0.00251, 0.00201, 0.00167, 0.00143, 0.00126, 0.00112, 0.00101])
        if connum > 10:
            avec = 0.01 / np.arange(11, connum + 1)
            dvec = np.concatenate((dvec, avec))
    else:
        dvec = alpha / np.arange(1, connum + 1)
        dvec[1] = alpha / 2
    if hoch:
        dvec = alpha / (2 * np.arange(1, connum + 1))
    dvec = 2 * dvec
    
    if plotit and connum == 1:
        plt.plot(np.concatenate((psihat[0, :], [0])), label="Est. Difference")
        plt.axhline(0, color='red')
        plt.show()
    
    temp2 = np.argsort(-test)
    zvec = dvec[:connum]
    sigvec = test[temp2] >= zvec
    output = np.zeros((connum, 6))
    tmeans = np.apply_along_axis(est, 1, xx, *args)
    icl = int(round(dvec[connum - 1] * nboot / 2)) + 1
    icu = nboot - icl - 1
    for ic in range(connum):
        output[ic, 1] = ic + 1
        output[ic, 2] = tmeans[ic]
        output[ic, 3] = test[ic]
        output[temp2, 4] = zvec
        temp = np.sort(psihat[ic, :])
        output[ic, 5] = temp[icl]
        output[ic, 6] = temp[icu]
    num_sig = np.sum(output[:, 3] <= output[:, 4])
    return {"output": output, "con": con, "num.sig": num_sig}


def HuberTun(kappa, p):
    prob = 1 - kappa
    chip = chi2.ppf(prob, p)
    r = np.sqrt(chip)
    tau = (p * chi2.cdf(chip, p + 2) + chip * (1 - prob)) / p
    Results = {'r': r, 'tau': tau}
    return Results

def robEst(Z, r, tau, ep):
    p = Z.shape[1]
    n = Z.shape[0]
    
    mu0 = MeanCov(Z)['zbar']
    Sigma0 = MeanCov(Z)['S']
    Sigin = np.linalg.inv(Sigma0)
    diverg = 0 
    for k in range(1, 201):
        sumu1 = 0
        mu = np.zeros((p, 1))
        Sigma = np.zeros((p, p))
        d = np.empty(n)
        u1 = np.empty(n)
        u2 = np.empty(n)
        for i in range(n):
            zi = Z[i,:]
            zi0 = zi - mu0
            di2 = np.dot(np.dot(zi0.T, Sigin), zi0)
            di = np.sqrt(di2)
            d[i] = di

            if di <= r:
                u1i = 1.0
                u2i = 1.0 / tau
            else:
                u1i = r / di
                u2i = u1i ** 2 / tau
            u1[i] = u1i
            u2[i] = u2i
            sumu1 = sumu1 + u1i
            mu = mu + u1i * zi
            Sigma = Sigma + u2i * np.dot(zi0, zi0.T)

        mu1 = mu / sumu1
        Sigma1 = Sigma / n
        Sigdif = Sigma1 - Sigma0
        dt = np.sum(Sigdif ** 2)
        mu0 = mu1
        Sigma0 = Sigma1
        Sigin = np.linalg.inv(Sigma0)
        if dt < ep:
            break

    if k == 200:
        diverg = 1
        mu0 = np.zeros(p)
        sigma0 = np.empty((p, p))
    theta = MLEst(Sigma0)
    Results = {'mu': mu0, 'Sigma': Sigma0, 'theta': theta, 'd': d, 'u1': u1, 'u2': u2, 'diverg': diverg}
    return Results

def SErob(Z, mu, Sigma, theta, d, r, tau):
    n = Z.shape[0]
    p = Z.shape[1]
    ps = p * (p + 1) / 2
    q = 6
    Dup = Dp(p)
    DupPlus = np.dot(np.linalg.inv(np.dot(Dup.T, Dup)), Dup.T)
    InvSigma = np.linalg.inv(Sigma)
    sigma = vech(Sigma)
    W = 0.5 * np.dot(np.dot(Dup.T, np.kron(InvSigma, InvSigma)), Dup)
    Zr = np.empty((n, p))
    A = np.zeros((p + q, p + q))
    B = np.zeros((p + q, p + q))
    sumg = np.zeros(p + q)
    for i in range(n):
        zi = Z[i,:]
        zi0 = zi - mu
        di = d[i]
        if di <= r:
            u1i = 1.0
            u2i = 1.0 / tau
            du1i = 0
            du2i = 0
        else:
            u1i = r / di
            u2i = u1i ** 2 / tau
            du1i = -r / di ** 2
            du2i = -2 * r ** 2 / tau / di ** 3

        Zr[i,:] = np.sqrt(u2i) * zi0

        g1i = u1i * zi0 
        vTi = vech(np.dot(zi0, zi0.T))
        g2i = u2i * vTi - sigma 
        gi = np.vstack((g1i, g2i))
        sumg = gi + sumg
        B = B + np.dot(gi, gi.T)


        ddmu = -1 / di * np.dot(zi0.T, InvSigma)
        ddsigma = -np.dot(vTi.T, W) / di

        dg1imu = -u1i * np.eye(p) + du1i * np.dot(zi0, ddmu)
        dg1isigma = du1i * np.dot(zi0, ddsigma)
        dg2imu = -u2i * np.dot(DupPlus, np.kron(zi0, np.eye(p)) + np.kron(np.eye(p), zi0)) + du2i * np.dot(vTi, ddmu)
        dg2isigma = du2i * np.dot(vTi, ddsigma) - np.eye(q)
        dgi = np.vstack((np.hstack((dg1imu, dg1isigma)), np.hstack((dg2imu, dg2isigma))))
        A = A + dgi

    A = -1 * A / n
    B = B / n
    invA = np.linalg.inv(A)
    OmegaSW = np.dot(np.dot(invA, B), invA.T)
    OmegaSW = OmegaSW[(p + 1):(p + q), (p + 1):(p + q)]
    SEsw = getSE(theta, OmegaSW, n)
    SEinf = SEML(Zr, theta)['inf']
    Results = {'inf': SEinf, 'sand': SEsw, 'Zr': Zr}
    return Results

def MeanCov(Z):
    n = Z.shape[0]
    p = Z.shape[1]
    zbar = np.dot(Z.T, np.ones((n, 1))) / n
    S = np.dot(np.dot(Z.T, np.eye(n) - np.ones((n, n))) , Z) / n
    Results = {'zbar': zbar, 'S': S}
    return Results

def MLEst(S):
    ahat = S[0, 1] / S[0, 0]
    vx = S[0, 0]

    Sxx = S[0:2, 0:2]
    sxy = S[0:2, 2]
    vem = S[1, 1] - S[1, 0] * S[0, 1] / S[0, 0]

    invSxx = np.linalg.inv(Sxx)
    beta_v = np.dot(invSxx, sxy)
    vey = S[2, 2] - np.dot(np.dot(sxy.T, invSxx), sxy)
    thetaMLE = np.array([ahat, beta_v[1], beta_v[0], vx, vem, vey])
    return thetaMLE

def Dp(p):
    p2 = p * p
    ps = p * (p + 1) / 2
    Dup = np.zeros((p2, ps))
    count = 0
    for j in range(1, p + 1):
        for i in range(j, p + 1):
            count = count + 1
            if i == j:
                Dup[(j - 1) * p + j - 1, count - 1] = 1
            else:
                Dup[(j - 1) * p + i - 1, count - 1] = 1
                Dup[(i - 1) * p + j - 1, count - 1] = 1
    return Dup

def vech(A):
    l = 0
    p = A.shape[0]
    ps = p * (p + 1) / 2
    vA = np.zeros((ps, 1))
    for i in range(p):
        for j in range(i, p):
            l = l + 1
            vA[l - 1, 0] = A[j, i]
    return vA

def getSE(theta, Omega, n):
    hdot = gethdot(theta)
    COV = np.dot(np.dot(hdot, Omega / n), hdot.T)
    se_v = np.sqrt(np.diag(COV))
    a = theta[0]
    b = theta[1]
    SobelSE = np.sqrt(a ** 2 * COV[1, 1] + b ** 2 * COV[0, 0])
    se_v = np.append(se_v, SobelSE)
    return se_v

def gethdot(theta):
    p = 3
    ps = p * (p + 1) / 2
    q = ps
    a = theta[0]
    b = theta[1]
    c = theta[2]

    vx = theta[3]
    vem = theta[4]
    vey = theta[5]
    sigmadot = np.zeros((ps, q))
    sigmadot[0,:] = np.array([0, 0, 0, 1, 0, 0])
    sigmadot[1,:] = np.array([vx, 0, 0, a, 0, 0])
    sigmadot[2,:] = np.array([b * vx, a * vx, vx, a * b + c, 0, 0])
    sigmadot[3,:] = np.array([2 * a * vx, 0, 0, a ** 2, 1, 0])
    sigmadot[4,:] = np.array([(2 * a * b + c) * vx, a ** 2 * vx + vem, a * vx, a ** 2 * b + a * c, b, 0])
    sigmadot[5,:] = np.array([(2 * b * c + 2 * a * b ** 2) * vx, (2 * c * a + 2 * a ** 2 * b) * vx + 2 * b * vem, (2 * a * b + 2 * c) * vx, c ** 2 + 2 * c * a * b + a ** 2 * b ** 2, b ** 2, 1])
    hdot = np.linalg.inv(sigmadot)
    return hdot

def SEML(Z, thetaMLE):
    n = Z.shape[0]
    p = Z.shape[1]
    ps = p * (p + 1) / 2
    q = ps
    zbar = MeanCov(Z)['zbar']
    S = MeanCov(Z)['S']
    Dup = Dp(p)
    InvS = np.linalg.inv(S)
    W = 0.5 * np.dot(np.dot(Dup.T, np.kron(InvS, InvS)), Dup)
    OmegaInf = np.linalg.inv(W)

    S12 = np.zeros((p, ps))
    S22 = np.zeros((ps, ps))
    for i in range(n):
        zi0 = Z[i,:] - zbar
        difi = np.dot(zi0, zi0.T) - S
        vdifi = vech(difi)
        S12 = S12 + np.dot(zi0, vdifi.T)
        S22 = S22 + np.dot(vdifi, vdifi.T)
    OmegaSW = S22 / n
    SEinf = getSE(thetaMLE, OmegaInf, n)
    SEsw = getSE(thetaMLE, OmegaSW, n)
    Results = {'inf': SEinf, 'sand': SEsw}
    return Results

def BCI(Z, Zr, abH, B, level, ab = None):
    p = Z.shape[1]
    n = Z.shape[0]

    abhatH_v = np.empty(B)
    Index_m = np.empty((n, B))
    t1 = 0
    t2 = 0
    for i in range(B):
        U = np.random.uniform(1, n + 1, n)
        index = np.floor(U).astype(int)
        Index_m[:, i] = index

        Zrb = Zr[index - 1,:]
        SH = MeanCov(Zrb)['S']
        thetaH = MLEst(SH)
        abhatH = thetaH[0] * thetaH[1]
        abhatH_v[i] = abhatH
        if abhatH < abH:
            t2 = t2 + 1

    abhatH_v = abhatH_v[~np.isnan(abhatH_v)]
    SEBH = np.std(abhatH_v)
    CI2 = BpBCa(Zr, abhatH_v, t2, level)
    Results = {'CI': CI2[0], 'pv': CI2[1]}
    return Results

def BpBCa(Z, abhat_v, t, level):
    oab_v = np.sort(abhat_v)
    B = len(abhat_v)
    ranklowBp = round(B * level / 2)
    if ranklowBp == 0:
        ranklowBp = 1
    Bpl = oab_v[ranklowBp - 1]
    Bph = oab_v[round(B * (1 - level / 2)) - 1]
    BP = np.array([Bpl, Bph])
    pstar = np.mean(oab_v > 0)
    pv = 2 * min(pstar, 1 - pstar)
    return [BP, pv]

def idealf(x, na_rm = False):
    if na_rm:
        x = x[~np.isnan(x)]
    j = np.floor(len(x) / 4 + 5 / 12).astype(int)
    y = np.sort(x)
    g = (len(x) / 4) - j + (5 / 12)
    ql = (1 - g) * y[j - 1] + g * y[j]
    k = len(x) - j + 1
    qu = (1 - g) * y[k - 1] + g * y[k - 2]
    return {'ql': ql, 'qu': qu}

def ifmest(x, bend = 1.28, op = 2):
    tt = mest(x, bend)
    s = np.median(np.abs(x - np.median(x))) * norm.ppf(.75)

    val = kerden(x, 0, tt)
    val1 = kerden(x, 0, tt - s)
    val2 = kerden(x, 0, tt + s)

    ifmad = np.sign(np.abs(x - tt) - s) - (val2 - val1) * np.sign(x - tt) / val
    ifmad = ifmad / (2 * .6745 * (val2 + val1))
    y = (x - tt) / np.median(np.abs(x - np.median(x)))
    n = len(x)
    b = np.sum(y[np.abs(y) <= bend]) / n
    a = hpsi(y, bend) * np.median(np.abs(x - np.median(x))) - ifmad * b
    ifmest = a / (len(y[np.abs(y) <= bend]) / n)
    return ifmest

def kerden(x, q = .5, xval = 0):
    y = np.sort(x)
    n = len(x)
    temp = idealf(x)
    h = 1.2 * (temp['qu'] - temp['ql']) / n ** (.2)
    iq = np.floor(q * n + .5).astype(int)
    qhat = y[iq - 1]
    if q == 0:
        qhat = xval
    xph = qhat + h
    A = len(y[y <= xph])
    xmh = qhat - h
    B = len(y[y < xmh])
    fhat = (A - B) / (2 * n * h)
    return fhat

def hpsi(x, bend = 1.28):
    hpsi = np.where(np.abs(x) <= bend, x, bend * np.sign(x))
    return hpsi

def yuenv2(x, y = None, tr = .2, alpha = .05, plotit = False, op = True, VL = True, cor_op = False, loc_fun = np.median,
           xlab = "Groups", ylab = "", PB = False, nboot = 100, SEED = False, **kwargs):
    if y is None:
        if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame):
            y = x[:, 1]
            x = x[:, 0]
        if isinstance(x, list):
            y = x[1]
            x = x[0]

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1 = len(x)
    n2 = len(y)
    h1 = len(x) - 2 * np.floor(tr * len(x))
    h2 = len(y) - 2 * np.floor(tr * len(y))
    q1 = (len(x) - 1) * winvar(x, tr) / (h1 * (h1 - 1))
    q2 = (len(y) - 1) * winvar(y, tr) / (h2 * (h2 - 1))
    df = (q1 + q2) ** 2 / ((q1 ** 2 / (h1 - 1)) + (q2 ** 2 / (h2 - 1)))
    crit = norm.ppf(1 - alpha / 2)
    m1 = np.mean(x, tr)
    m2 = np.mean(y, tr)
    mbar = (m1 + m2) / 2
    dif = m1 - m2
    low = dif - crit * np.sqrt(q1 + q2)
    up = dif + crit * np.sqrt(q1 + q2)
    test = np.abs(dif / np.sqrt(q1 + q2))
    yuen = 2 * (1 - norm.cdf(test))
    xx = np.concatenate((np.repeat(1, len(x)), np.repeat(2, len(y))))
    if h1 == h2:
        pts = np.concatenate((x, y))
        top = np.var(np.concatenate((m1, m2)))

        if not PB:
            if tr == 0:
                cterm = 1
            if tr > 0:
                cterm = np.array(dnormvar, norm.ppf(tr), norm.ppf(1 - tr)) + 2 * (norm.ppf(tr) ** 2) * tr
            bot = winvar(pts, tr = tr) / cterm
            e_pow = top / bot
            if e_pow > 1:
                x0 = np.concatenate((np.repeat(1, len(x)), np.repeat(2, len(y))))
                y0 = np.concatenate((x, y))
                e_pow = wincor(x0, y0, tr = tr)['cor'] ** 2
    if n1 != n2:
        N = min(n1, n2)
        vals = np.zeros(nboot)
        for i in range(nboot):
            vals[i] = yuen.effect(np.random.choice(x, N), np.random.choice(y, N), tr = tr)['Var.Explained']
        e_pow = loc_fun(vals)
    return {'ci': np.array([low, up]), 'n1': n1, 'n2': n2, 'p.value': yuen, 'dif': dif, 'se': np.sqrt(q1 + q2),
            'teststat': test, 'crit': crit, 'df': df, 'Var.Explained': e_pow, 'Effect.Size': np.sqrt(e_pow)}


def yuen_effect(x, y, tr=0.2, alpha=0.05, plotit=False, op=True, VL=True, cor_op=False, xlab="Groups", ylab="", PB=False, **kwargs):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    h1 = len(x) - 2 * np.floor(tr * len(x))
    h2 = len(y) - 2 * np.floor(tr * len(y))
    q1 = (len(x) - 1) * winvar(x, tr) / (h1 * (h1 - 1))
    q2 = (len(y) - 1) * winvar(y, tr) / (h2 * (h2 - 1))
    df = (q1 + q2) ** 2 / ((q1 ** 2 / (h1 - 1)) + (q2 ** 2 / (h2 - 1)))
    crit = t.ppf(1 - alpha / 2, df)
    m1 = np.mean(x, tr)
    m2 = np.mean(y, tr)
    mbar = (m1 + m2) / 2
    dif = m1 - m2
    low = dif - crit * np.sqrt(q1 + q2)
    up = dif + crit * np.sqrt(q1 + q2)
    test = np.abs(dif / np.sqrt(q1 + q2))
    yuen = 2 * (1 - t.cdf(test, df))
    xx = np.concatenate((np.repeat(1, len(x)), np.repeat(2, len(y))))
    pts = np.concatenate((x, y))
    top = np.var(np.concatenate((m1, m2)))
    
    if not PB:
        if tr == 0:
            cterm = 1
        if tr > 0:
            cterm = np.array(dnormvar, norm.ppf(tr), norm.ppf(1 - tr)) + 2 * (norm.ppf(tr) ** 2) * tr
        bot = winvar(pts, tr=tr) / cterm
    
    e_pow = top / bot
    if e_pow > 1:
        x0 = np.concatenate((np.repeat(1, len(x)), np.repeat(2, len(y))))
        y0 = np.concatenate((x, y))
        e_pow = wincor(x0, y0, tr=tr)['cor'] ** 2
    
    return {'ci': [low, up], 'p.value': yuen, 'dif': dif, 'se': np.sqrt(q1 + q2),
            'teststat': test, 'crit': crit, 'df': df,
            'Var.Explained': e_pow, 'Effect.Size': np.sqrt(e_pow)}

def D_akp_effect(x, y=None, null_value=0, tr=0.2):
    if y is not None:
        x = x - y
    x = elimna(x)
    s1sq = winvar(x, tr=tr)
    cterm = 1
    if tr > 0:
        cterm = np.array(dnormvar, norm.ppf(tr), norm.ppf(1 - tr)) + 2 * (norm.ppf(tr) ** 2) * tr
    cterm = np.sqrt(cterm)
    dval = cterm * (tmean(x, tr=tr) - null_value) / np.sqrt(s1sq)
    return dval

def D_akp_effect_ci(x, y=None, null_value=0, alpha=0.05, tr=0.2, nboot=1000, SEED=False):
    if SEED:
        np.random.seed(2)
    if y is not None:
        x = x - y
    x = elimna(x)
    a = D_akp_effect(x=x, tr=tr, null_value=null_value)
    v = np.empty(nboot)
    for i in range(nboot):
        X = np.random.choice(x, replace=True)
        v[i] = D_akp_effect(X, tr=tr, null_value=null_value)
    v = np.sort(v)
    ilow = round((alpha / 2) * nboot)
    ihi = nboot - ilow
    ilow = ilow + 1
    ci = [v[ilow], v[ihi]]
    return {'Effect.Size': a, 'ci': ci}

def depQS(x, y=None, locfun=np.median, **kwargs):
    if y is not None:
        L = x - y
    else:
        L = x
    L = elimna(L)
    est = locfun(L, **kwargs)
    if est >= 0:
        ef_sizeND = np.mean(L - est <= est)
    ef_size = np.mean(L - est <= est)
    return {'Q.effect': ef_size}

def depQSci(x, y=None, locfun=np.median, alpha=0.05, nboot=1000, SEED=False, **kwargs):
    if SEED:
        np.random.seed(2)
    if y is not None:
        xy = elimna(np.column_stack((x, y)))
        xy = xy[:, 0] - xy[:, 1]
    if y is None:
        xy = elimna(x)
    n = len(xy)
    v = np.empty(nboot)
    for i in range(nboot):
        id = np.random.choice(np.arange(1, n+1), replace=True)
        v[i] = depQS(xy[id], locfun=locfun, **kwargs)['Q.effect']
    v = np.sort(v)
    ilow = round((alpha / 2) * nboot)
    ihi = nboot - ilow
    ilow = ilow + 1
    ci = [v[ilow], v[ihi]]
    est = depQS(xy, **kwargs)['Q.effect']
    return {'Q.effect': est, 'ci': ci}

def binom_conf(y, AUTO=True, method='SD', n=np.nan, alpha=0.05):
    method = 'SD'
    x, nn = sum(y), len(y)
    if nn < 35:
        if AUTO:
            method = 'SD'
    type = method
    if type == 'SD':
        return binomLCO(x=x, nn=nn, y=y, alpha=alpha)

def binomLCO(y=None, n=np.nan, alpha=0.05):
    if y is not None:
        y = elimna(y)
        nn = len(y)
    else:
        x, nn = sum(y), len(y)
    if nn == 1:
        raise ValueError('Something is wrong: number of observations is only 1')
    cis = LCO_CI(nn, 1 - alpha, 3)
    ci = cis[x + 1, 1:2]
    return {'phat': x / nn, 'ci': ci, 'n': nn}

def LCO_CI(n, level, dp):
    iter = 10 ** (dp + 1)
    p = np.arange(0, 0.5, 1 / iter)
    cpf_matrix = np.empty((iter + 1, 3))
    cpf_matrix[:, 0] = p
    for i in range(1, int(iter / 2) + 2):
        p = (i - 1) / iter
        bin = binomLCO.pmf(np.arange(0, n+1), n, p)
        x = np.arange(0, n+1)
        pmf = np.column_stack((x, bin))
        pmf = pmf[np.lexsort((-pmf[:, 1], pmf[:, 0])), :]
        m_row = np.min(np.where(np.cumsum(pmf[:, 1]) >= level))
        low_val = np.min(pmf[0:m_row, 0])
        upp_val = np.max(pmf[0:m_row, 0])
        cpf_matrix[i-1, 1:3] = [low_val, upp_val]
        if i != int(iter / 2) + 1:
            n_p = 1 - p
            n_low = n - upp_val
            n_upp = n - low_val
            cpf_matrix[iter + 1 - i, 1:3] = [n_low, n_upp]
    diff_l = np.diff(cpf_matrix[:, 1])
    if np.min(diff_l) == -1:
        for i in np.where(diff_l == -1)[0]:
            j = np.min(np.where(diff_l == 1)[np.where(diff_l == 1) > i])
            new_low = cpf_matrix[j, 1]
            new_upp = cpf_matrix[j, 2]
            cpf_matrix[i:j, 1] = new_low
            cpf_matrix[i:j, 2] = new_upp
        pointer_1 = iter - (j - 1) + 2
        pointer_2 = iter - i + 2
        cpf_matrix[pointer_1:pointer_2, 1] = n - new_upp
        cpf_matrix[pointer_1:pointer_2, 2] = n - new_low
    ci_matrix = np.empty((n + 1, 3))
    ci_matrix[:, 0] = np.arange(0, n+1)
    for x in range(0, int(n/2)+1):
        num_row = np.sum((cpf_matrix[:, 1] <= x) & (x <= cpf_matrix[:, 2]))
        low_lim = np.round(cpf_matrix[(cpf_matrix[:, 1] <= x) & (x <= cpf_matrix[:, 2]), 0][0], dp)
        upp_lim = np.round(cpf_matrix[(cpf_matrix[:, 1] <= x) & (x <= cpf_matrix[:, 2]), 0][num_row-1], dp)
        ci_matrix[x, 1:3] = [low_lim, upp_lim]
        n_x = n - x
        n_low_lim = 1 - upp_lim
        n_upp_lim = 1 - low_lim
        ci_matrix[n_x, 1:3] = [n_low_lim, n_upp_lim]
    return ci_matrix

def rmES_pro(x, est=tmean, **kwargs):
    if isinstance(x, list):
        x = matl(x)
    x = elimna(x)
    E = np.apply_along_axis(est, 0, x, **kwargs)
    GM = np.mean(E)
    J = x.shape[1]
    GMvec = np.repeat(GM, J)
    DN = pdis(x, GMvec, center=E)
    return DN

def pdis(m, MM=False, cop=3, dop=1, center=np.nan, na_rm=True):
    m = elimna(m)
    pts = elimna(pts)
    m = np.asarray(m)
    nm = m.shape[0]
    pts = np.asarray(pts)
    if m.shape[1] > 1:
        if pts.shape[1] == 1:
            pts = pts.T
    npts = pts.shape[0]
    mp = np.vstack((m, pts))
    np1 = m.shape[0] + 1
    if m.shape[1] == 1:
        m = np.ravel(m)
        pts = np.ravel(pts)
        if np.isnan(center[0]):
            center = np.median(m)
        dis = np.abs(pts - center)
        disall = np.abs(m - center)
        temp = idealf(disall)
        if not MM:
            pdis = dis / (temp['qu'] - temp['ql'])
        if MM:
            pdis = dis / mad(disall)
    else:
        if np.isnan(center[0]):
            if cop == 1:
                center = dmean(m, tr=0.5, dop=dop)
            if cop == 2:
                center = cov.mcd(m)['center']
            if cop == 3:
                center = np.apply_along_axis(np.median, 0, m)
            if cop == 4:
                center = cov.mve(m)['center']
            if cop == 5:
                center = smean(m)
        dmat = np.empty((mp.shape[0], mp.shape[0]))
        for i in range(mp.shape[0]):
            B = mp[i, :] - center
            dis = np.empty(mp.shape[0])
            BB = B ** 2
            bot = np.sum(BB)
            if bot != 0:
                for j in range(mp.shape[0]):
                    A = mp[j, :] - center
                    temp = np.sum(A * B) * B / bot
                    dis[j] = np.sqrt(np.sum(temp ** 2))
                dis_m = dis[:nm]
                if not MM:
                    temp = idealf(dis_m)
                    dmat[:, i] = dis / (temp['qu'] - temp['ql'])
                if MM:
                    dmat[:, i] = dis / mad(dis_m)
    pdis = np.apply_along_axis(np.max, 1, dmat, na_rm=na_rm)
    pdis = pdis[np1:]
    return pdis

