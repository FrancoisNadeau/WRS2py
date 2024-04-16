def con2way(J, K):
    JK = J * K
    Ja = (J**2 - J) / 2
    Ka = (K**2 - K) / 2
    JK = J * K
    conA = [[0] * Ja for _ in range(JK)]
    ic = 0
    for j in range(1, J+1):
        for jj in range(1, J+1):
            if j < jj:
                ic += 1
                mat = [[0] * K for _ in range(J)]
                mat[j-1] = [1] * K
                mat[jj-1] = [-1] * K
                conA[:, ic-1] = np.transpose(mat)
    conB = [[0] * Ka for _ in range(JK)]
    ic = 0
    for k in range(1, K+1):
        for kk in range(1, K+1):
            if k < kk:
                ic += 1
                mat = [[0] * K for _ in range(J)]
                for i in range(J):
                    mat[i][k-1] = 1
                    mat[i][kk-1] = -1
                conB[:, ic-1] = np.transpose(mat)
    conAB = [[0] * (Ka * Ja) for _ in range(JK)]
    ic = 0
    for j in range(1, J+1):
        for jj in range(1, J+1):
            if j < jj:
                for k in range(1, K+1):
                    for kk in range(1, K+1):
                        if k < kk:
                            ic += 1
                            mat = [[0] * K for _ in range(J)]
                            mat[j-1][k-1] = 1
                            mat[j-1][kk-1] = -1
                            mat[jj-1][k-1] = -1
                            mat[jj-1][kk-1] = 1
                            conAB[:, ic-1] = np.transpose(mat)
    return {'conA': conA, 'conB': conB, 'conAB': conAB}