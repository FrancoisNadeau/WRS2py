def matl(x):
    J = len(x)
    nval = []
    for j in range(J):
        nval.append(len(x[j]))
    temp = [[None] * J for _ in range(max(nval))]
    for j in range(J):
        temp[:nval[j], j] = x[j]
    return temp