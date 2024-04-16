def winmean(x, tr=0.2, na_rm=False, *args, **kwargs):
    if na_rm:
        x = elimna(x)
    winmean = mean(winval(x, tr))
    return winmean