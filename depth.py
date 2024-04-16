import math

def depth(U, V, m):
    X = [row[0] for row in m]
    Y = [row[1] for row in m]
    FV = []
    NUMS = 0
    NUMH = 0
    SDEP = 0.0
    HDEP = 0.0
    N = len(X)
    P = math.acos(-1)
    P2 = P * 2.0
    EPS = 0.000001
    ALPHA = []
    NT = 0
    
    for i in range(len(m)):
        DV = math.sqrt(((X[i] - U) * (X[i] - U) + (Y[i] - V) * (Y[i] - V)))
        if DV <= EPS:
            NT += 1
        else:
            XU = (X[i] - U) / DV
            YU = (Y[i] - V) / DV
            if abs(XU) > abs(YU):
                if X[i] >= U:
                    ALPHA.append(math.asin(YU))
                    if ALPHA[-1] < 0.0:
                        ALPHA[-1] = P2 + ALPHA[-1]
                else:
                    ALPHA.append(P - math.asin(YU))
            else:
                if Y[i] >= V:
                    ALPHA.append(math.acos(XU))
                else:
                    ALPHA.append(P2 - math.acos(XU))
                if ALPHA[-1] >= P2 - EPS:
                    ALPHA[-1] = 0.0
    
    NN = N - NT
    if NN <= 1:
        NUMS = NUMS + depths1(NT, 1) * depths1(NN, 2) + depths1(NT, 2) * depths1(NN, 1) + depths1(NT, 3)
        if N >= 3:
            SDEP = (NUMS + 0.0) / (depths1(N, 3) + 0.0)
        NUMH = NUMH + NT
        HDEP = (NUMH + 0.0) / (N + 0.0)
        return HDEP
    
    ALPHA = sorted(ALPHA[:NN])
    ANGLE = ALPHA[0] - ALPHA[NN - 1] + P2
    for i in range(1, NN):
        ANGLE = max(ANGLE, ALPHA[i] - ALPHA[i - 1])
    
    if ANGLE > (P + EPS):
        NUMS = NUMS + depths1(NT, 1) * depths1(NN, 2) + depths1(NT, 2) * depths1(NN, 1) + depths1(NT, 3)
        if N >= 3:
            SDEP = (NUMS + 0.0) / (depths1(N, 3) + 0.0)
        NUMH = NUMH + NT
        HDEP = (NUMH + 0.0) / (N + 0.0)
        return HDEP
    
    ANGLE = ALPHA[0]
    NU = 0
    for i in range(NN):
        ALPHA[i] = ALPHA[i] - ANGLE
        if ALPHA[i] < (P - EPS):
            NU += 1
    
    if NU >= NN:
        NUMS = NUMS + depths1(NT, 1) * depths1(NN, 2) + depths1(NT, 2) * depths1(NN, 1) + depths1(NT, 3)
        if N >= 3:
            SDEP = (NUMS + 0.0) / (depths1(N, 3) + 0.0)
        NUMH = NUMH + NT
        HDEP = (NUMH + 0.0) / (N + 0.0)
        return HDEP
    
    JA = 1
    JB = 1
    ALPHK = ALPHA[0]
    BETAK = ALPHA[NU + 1] - P
    NN2 = NN * 2
    NBAD = 0
    I = NU
    NF = NN
    
    for J in range(1, NN2):
        ADD = ALPHK + EPS
        if ADD < BETAK:
            NF = NF + 1
            if JA < NN:
                JA = JA + 1
                ALPHK = ALPHA[JA]
            else:
                ALPHK = P2 + 1.0
        else:
            I = I + 1
            NN1 = NN + 1
            if I == NN1:
                I = 1
                NF = NF - NN
            FV[I] = NF
            NFI = NF - I
            NBAD = NBAD + depths1(NFI, 2)
            if JB < NN:
                JB = JB + 1
                if JB + NU <= NN:
                    BETAK = ALPHA[JB + NU] - P
                else:
                    BETAK = ALPHA[JB + NU - NN] + P
            else:
                BETAK = P2 + 1.0
    
    NUMS = depths1(NN, 3) - NBAD
    
    GI = 0
    JA = 1
    ANGLE = ALPHA[0]
    dif = NN - FV[0]
    NUMH = min(FV[0], dif)
    
    for I in range(1, NN):
        AEPS = ANGLE + EPS
        if ALPHA[I] <= AEPS:
            JA = JA + 1
        else:
            GI = GI + JA
            JA = 1
            ANGLE = ALPHA[I]
        KI = FV[I] - GI
        NNKI = NN - KI
        NUMH = min(NUMH, min(KI, NNKI))
    
    NUMS = NUMS + depths1(NT, 1) * depths1(NN, 2) + depths1(NT, 2) * depths1(NN, 1) + depths1(NT, 3)
    if N >= 3:
        SDEP = (NUMS + 0.0) / (depths1(N, 3) + 0.0)
    NUMH = NUMH + NT
    HDEP = (NUMH + 0.0) / (N + 0.0)
    return HDEP