def pball(x, beta=0.2, *args, **kwargs):
    cl = locals()
    m = x
    pbcorm = [[0] * len(m[0]) for _ in range(len(m[0]))]
    temp = [[1] * len(m[0]) for _ in range(len(m[0]))]
    siglevel = [[None] * len(m[0]) for _ in range(len(m[0]))]
    cmat = [[0] * len(m[0]) for _ in range(len(m[0]))]
    for i in range(len(m[0])):
        ip1 = i
        for j in range(ip1, len(m[0])):
            if i < j:
                pbc = pbcor(m[:, i], m[:, j], beta)
                pbcorm[i][j] = pbc['cor']
                temp[i][j] = pbcorm[i][j]
                temp[j][i] = pbcorm[i][j]
                siglevel[i][j] = pbc['p.value']
                siglevel[j][i] = siglevel[i][j]
    tstat = [[pbcorm[i][j] * ((len(m) - 2) / (1 - pbcorm[i][j] ** 2)) for j in range(len(m[0]))] for i in range(len(m[0]))]
    cmat = [[(tstat[i][j] * sqrt((len(m) - 2.5) * log(1 + tstat[i][j] ** 2 / (len(m) - 2)))) for j in range(len(m[0]))] for i in range(len(m[0]))]
    bv = 48 * (len(m) - 2.5) ** 2
    cmat = [[(cmat[i][j] + (cmat[i][j] ** 3 + 3 * cmat[i][j]) / bv - (4 * cmat[i][j] ** 7 + 33 * cmat[i][j] ** 5 + 240 ** cmat[i][j] ** 3 + 855 * cmat[i][j]) / (10 * bv ** 2 + 8 * bv * cmat[i][j] ** 4 + 1000 * bv)) for j in range(len(m[0]))] for i in range(len(m[0]))]
    H = sum([cmat[i][j] ** 2 for i in range(len(m[0])) for j in range(len(m[0]))])
    df = len(m[0]) * (len(m[0]) - 1) / 2
    h_siglevel = 1 - pchisq(H, df)
    if x.columns is not None:
        temp = siglevel = x.columns
    result = {'pbcorm': temp, 'p.values': siglevel, 'H': H, 'H.p.value': h_siglevel, 'call': cl}
    result.__class__.__name__ = 'pball'
    return result