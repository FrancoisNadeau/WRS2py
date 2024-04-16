def mest(x, bend=1.28, na_rm=False, *args, **kwargs):
    if na_rm:
        x = [i for i in x if i is not None]
    if mad(x) == 0:
        raise ValueError("MAD=0. The M-estimator cannot be computed.")
    y = [(i - median(x)) / mad(x) for i in x]
    A = sum(hpsi(y, bend))
    B = len([i for i in x if abs(y[i]) <= bend])
    mest = median(x) + mad(x) * A / B
    while True:
        y = [(i - mest) / mad(x) for i in x]
        A = sum(hpsi(y, bend))
        B = len([i for i in x if abs(y[i]) <= bend])
        newmest = mest + mad(x) * A / B
        if abs(newmest - mest) < 0.0001:
            break
        mest = newmest
    return mest