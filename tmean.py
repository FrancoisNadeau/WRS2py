def tmean(x, tr=0.2, na_rm=False, STAND=None):
    if na_rm:
        x = [i for i in x if i is not None]
    val = sum(x) / len(x)
    return val