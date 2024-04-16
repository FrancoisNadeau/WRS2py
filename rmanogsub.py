def rmanogsub(isub, x, est=onestep, *args, **kwargs):
    tsub = est(x[isub], *args, **kwargs)
    return tsub