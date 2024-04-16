def tsubrmanovab(isub, x, tr):
    from rmanovab import rmanovab1
    
    tsub = rmanovab1(x[isub,], tr=tr)['test']
    return tsub