def idealf(x, na_rm=False):
    if na_rm:
        x = [i for i in x if i is not None]
    j = len(x) // 4 + 5 / 12
    y = sorted(x)
    g = len(x) / 4 - j + 5 / 12
    ql = (1 - g) * y[int(j)] + g * y[int(j) + 1]
    k = len(x) - int(j) + 1
    qu = (1 - g) * y[k] + g * y[k - 1]
    return {'ql': ql, 'qu': qu}