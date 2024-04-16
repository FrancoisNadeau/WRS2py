def smean(m, cop=3, MM=False, op=1, outfun=outogk, cov_fun=rmba, MC=False, STAND=False, *args, **kwargs):
    m = elimna(m)
    if op == 1:
        if not MC:
            temp = outpro(m, plotit=False, cop=cop, MM=MM, STAND=STAND)['keep']
    if op == 2:
        temp = outmgv(m, plotit=False, cov_fun=cov_fun)['keep']
    if op == 3:
        temp = outfun(m, plotit=False, *args, **kwargs)['keep']
    val = [mean(m[temp, col]) for col in range(m.shape[1])]
    return val