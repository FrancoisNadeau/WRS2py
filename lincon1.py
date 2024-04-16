import numpy as np
from scipy.stats import t

def lincon1(x, con=0, tr=0.2, alpha=0.05, pr=True, crit=np.nan, SEED=True, KB=False):
    if tr == 0.5:
        raise ValueError("Use the R function medpb to compare medians")
    if isinstance(x, pd.DataFrame):
        x = x.values
    if KB:
        raise ValueError("Use the function kbcon")
    flag = True
    if alpha != 0.05 and alpha != 0.01:
        flag = False
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, list):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    con = np.array(con)
    J = len(x)
    sam = np.nan
    h = np.zeros(J)
    w = np.zeros(J)
    xbar = np.zeros(J)
    for j in range(J):
        xx = [val for val in x[j] if not np.isnan(val)]
        x[j] = xx
        sam[j] = len(x[j])
        h[j] = len(x[j]) - 2 * np.floor(tr * len(x[j]))
        w[j] = ((len(x[j]) - 1) * winvar(x[j], tr)) / (h[j] * (h[j] - 1))
        xbar[j] = np.mean(x[j], tr)
    if np.sum(con**2) == 0:
        CC = (J**2 - J) / 2
        psihat = np.zeros((CC, 6))
        psihat[:, 0:2] = 0
        psihat[:, 3:6] = 0
        dimnames_psihat = [None, ["Group", "Group", "psihat", "ci.lower", "ci.upper", "p.value"]]
        test = np.zeros((CC, 6))
        test[:, 0:2] = 0
        test[:, 2:6] = np.nan
        dimnames_test = [None, ["Group", "Group", "test", "crit", "se", "df"]]
        jcom = 0
        for j in range(J):
            for k in range(J):
                if j < k:
                    jcom += 1
                    test[jcom, 2] = np.abs(xbar[j] - xbar[k]) / np.sqrt(w[j] + w[k])
                    sejk = np.sqrt(w[j] + w[k])
                    test[jcom, 4] = sejk
                    psihat[jcom, 0] = j
                    psihat[jcom, 1] = k
                    test[jcom, 0] = j
                    test[jcom, 1] = k
                    psihat[jcom, 2] = xbar[j] - xbar[k]
                    df = (w[j] + w[k])**2 / (w[j]**2 / (h[j] - 1) + w[k]**2 / (h[k] - 1))
                    test[jcom, 5] = df
                    psihat[jcom, 5] = 2 * (1 - t.cdf(test[jcom, 2], df))
                    if not KB:
                        if CC > 28:
                            flag = False
                        if flag:
                            if alpha == 0.05:
                                crit = smmcrit(df, CC)
                            if alpha == 0.01:
                                crit = smmcrit01(df, CC)
                        if not flag or CC > 28:
                            crit = smmvalv2(dfvec=np.repeat(df, CC), alpha=alpha, SEED=SEED)
                    if KB:
                        crit = np.sqrt((J - 1) * (1 + (J - 2) / df) * f.ppf(1 - alpha, J - 1, df))
                    test[jcom, 3] = crit
                    psihat[jcom, 3] = xbar[j] - xbar[k] - crit * sejk
                    psihat[jcom, 4] = xbar[j] - xbar[k] + crit * sejk
        return {"n": sam, "test": test, "psihat": psihat}
    if np.sum(con**2) > 0:
        if con.shape[0] != len(x):
            raise ValueError("The number of groups does not match the number of contrast coefficients.")
        psihat = np.zeros((con.shape[1], 5))
        psihat[:, 0] = np.arange(1, con.shape[1] + 1)
        psihat[:, 1] = np.sum(con * xbar, axis=0)
        test = np.zeros((con.shape[1], 5))
        test[:, 0] = np.arange(1, con.shape[1] + 1)
        df = 0
        for d in range(con.shape[1]):
            sejk = np.sqrt(np.sum(con[:, d]**2 * w))
            test[d, 1] = np.sum(con[:, d] * xbar) / sejk
            df = (np.sum(con[:, d]**2 * w))**2 / np.sum(con[:, d]**4 * w**2 / (h - 1))
            if flag:
                if alpha == 0.05:
                    crit = smmcrit(df, con.shape[1])
                if alpha == 0.01:
                    crit = smmcrit01(df, con.shape[1])
            if not flag:
                crit = smmvalv2(dfvec=np.repeat(df, con.shape[1]), alpha=alpha, SEED=SEED)
            test[d, 2] = crit
            test[d, 3] = sejk
            test[d, 4] = df
            psihat[d, 2] = psihat[d, 1] - crit * sejk
            psihat[d, 3] = psihat[d, 1] + crit * sejk
            psihat[d, 4] = 2 * (1 - t.cdf(np.abs(test[d, 1]), df))
        return {"n": sam, "test": test, "psihat": psihat}