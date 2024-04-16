def hpsi(x, bend=1.28):
    hpsi = x if abs(x) <= bend else bend * (x / abs(x))
    return hpsi