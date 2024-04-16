def listm(x):
    if x.ndim == 1:
        raise ValueError("The argument x must be a matrix or data frame")
    y = []
    for j in range(x.shape[1]):
        y.append(x[:, j])
    return y