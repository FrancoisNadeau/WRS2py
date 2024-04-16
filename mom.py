def mom(x, bend=2.24, na_rm=True, *args, **kwargs):
    if na_rm:
        x = [i for i in x if i is not None]
    median_x = median(x)
    mad_x = median([abs(i - median_x) for i in x])
    flag1 = [i > median_x + bend * mad_x for i in x]
    flag2 = [i < median_x - bend * mad_x for i in x]
    flag = [True] * len(x)
    for i in range(len(x)):
        if flag1[i] or flag2[i]:
            flag[i] = False
    mom = sum([x[i] for i in range(len(x)) if flag[i]]) / sum(flag)
    return mom