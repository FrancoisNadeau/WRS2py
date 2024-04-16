def twobinom(x, y, alpha=0.05):
    r1 = sum(x)
    n1 = len(x)
    r2 = sum(y)
    n2 = len(y)
    
    n1p = n1 + 1
    n2p = n2 + 1
    n1m = n1 - 1
    n2m = n2 - 1
    chk = abs(r1 / n1 - r2 / n2)
    x = [i / n1 for i in range(n1 + 1)]
    y = [i / n2 for i in range(n2 + 1)]
    phat = (r1 + r2) / (n1 + n2)
    m1 = [[x[i] - y[j] for j in range(n2 + 1)] for i in range(n1 + 1)]
    m2 = [[1] * n2p for _ in range(n1p)]
    flag = [[abs(m1[i][j]) >= chk for j in range(n2p)] for i in range(n1p)]
    m3 = [[m2[i][j] * flag[i][j] for j in range(n2p)] for i in range(n1p)]
    b1 = 1
    b2 = 1
    xv = [i + 1 for i in range(n1)]
    yv = [i + 1 for i in range(n2)]
    xv1 = [n1 - i for i in xv]
    yv1 = [n2 - i for i in yv]
    dis1 = [1] + [pbeta(phat, xv[i], xv1[i]) for i in range(n1)]
    dis2 = [1] + [pbeta(phat, yv[i], yv1[i]) for i in range(n2)]
    pd1 = [None] * (n1p + 1)
    pd2 = [None] * (n2p + 1)
    for i in range(n1):
        pd1[i] = dis1[i] - dis1[i + 1]
    for i in range(n2):
        pd2[i] = dis2[i] - dis2[i + 1]
    pd1[n1p] = phat ** n1
    pd2[n2p] = phat ** n2
    m4 = [[pd1[i] * pd2[j] for j in range(n2p)] for i in range(n1p)]
    test = sum([m3[i][j] * m4[i][j] for i in range(n1p) for j in range(n2p)])
    return {'p.value': test, 'p1': r1 / n1, 'p2': r2 / n2, 'est.dif': r1 / n1 - r2 / n2}