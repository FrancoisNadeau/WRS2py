import numpy as np
from scipy.stats import chi2

def rmba(x, csteps=5, na_rm=True, plotit=False):
    x = np.matrix(x)
    if na_rm:
        x = elimna(x)
    p = x.shape[1]
    n = x.shape[0]
    covs = np.cov(x, rowvar=False)
    mns = np.mean(x, axis=0)
    for i in range(csteps):
        md2 = mahalanobis(x, mns, covs)
        medd2 = np.median(md2)
        mns = np.mean(x[md2 <= medd2, :], axis=0)
        covs = np.cov(x[md2 <= medd2, :], rowvar=False)
    covb = covs
    mnb = mns
    critb = np.prod(np.diag(np.linalg.cholesky(covb)))
    covv = np.diag(np.ones(p))
    med = np.median(x, axis=0)
    md2 = mahalanobis(x, center=med, covv=covv)
    medd2 = np.median(md2)
    mns = np.mean(x[md2 <= medd2, :], axis=0)
    covs = np.cov(x[md2 <= medd2, :], rowvar=False)
    for i in range(csteps):
        md2 = mahalanobis(x, mns, covs)
        medd2 = np.median(md2)
        mns = np.mean(x[md2 <= medd2, :], axis=0)
        covs = np.cov(x[md2 <= medd2, :], rowvar=False)
    crit = np.prod(np.diag(np.linalg.cholesky(covs)))
    if crit < critb:
        critb = crit
        covb = covs
        mnb = mns
    rd2 = mahalanobis(x, mnb, covb)
    const = np.median(rd2) / (chi2.ppf(0.5, p))
    covb = const * covb
    rd2 = mahalanobis(x, mnb, covb)
    up = chi2.ppf(0.975, p)
    rmnb = np.mean(x[rd2 <= up, :], axis=0)
    rcovb = np.cov(x[rd2 <= up, :], rowvar=False)
    rd2 = mahalanobis(x, rmnb, rcovb)
    const = np.median(rd2) / (chi2.ppf(0.5, p))
    rcovb = const * rcovb
    rd2 = mahalanobis(x, rmnb, rcovb)
    up = chi2.ppf(0.975, p)
    rmnb = np.mean(x[rd2 <= up, :], axis=0)
    rcovb = np.cov(x[rd2 <= up, :], rowvar=False)
    rd2 = mahalanobis(x, rmnb, rcovb)
    const = np.median(rd2) / (chi2.ppf(0.5, p))
    rcovb = const * rcovb
    cor_b = None
    temp = np.outer(np.sqrt(np.diag(rcovb)), np.sqrt(np.diag(rcovb)))
    if np.min(np.diag(rcovb)) > 0:
        cor_b = rcovb / temp
    return {'center': rmnb, 'cov': rcovb, 'cor': cor_b}