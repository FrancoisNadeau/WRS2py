def selby(m, grpc, coln):
    if not isinstance(m, (list, tuple)) and not isinstance(m, np.ndarray):
        raise ValueError("Data must be stored in a matrix or data frame")
    if grpc[0] is None:
        raise ValueError("The argument grpc is not specified")
    if coln[0] is None:
        raise ValueError("The argument coln is not specified")
    if len(grpc) != 1:
        raise ValueError("The argument grpc must have length 1")
    x = []
    grpn = sorted(list(set(m[:, grpc[0]])))
    it = 0
    for ig in range(len(grpn)):
        for ic in range(len(coln)):
            it += 1
            flag = (m[:, grpc[0]] == grpn[ig])
            x.append(m[flag, coln[ic]])
    return {'x': x, 'grpn': grpn}