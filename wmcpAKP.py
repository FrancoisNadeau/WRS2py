import numpy as np

def wmcpAKP(x, tr=0.2, nboot=200, *args, **kwargs):
    if isinstance(x, (np.matrix, np.ndarray)):
        x = [x.tolist()]
    J = len(x)
    C = (J**2 - J) / 2
    A = np.empty((C, 5))
    A[:] = np.nan
    ic = 0
    for j in range(J):
        for k in range(J):
            if j < k:
                ic += 1
                A[ic-1, 0] = j+1
                A[ic-1, 1] = k+1
                dep_effect = dep_effect(x[j], x[k], tr=tr, nboot=nboot)
                A[ic-1, 2] = dep_effect[4]
                A[ic-1, 3] = dep_effect[20]
                A[ic-1, 4] = dep_effect[24]
    res = {"Effect.Size": np.mean(A[:, 2]), "ci.low": np.mean(A[:, 3]), "ci.up": np.mean(A[:, 4])}
    res = np.array(res)
    res = res.astype([("Effect.Size", float), ("ci.low", float), ("ci.up", float)])
    return res