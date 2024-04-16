def pbos(x, beta=0.2):
    temp = sorted([abs(i - median(x)) for i in x])
    omhatx = temp[int((1 - beta) * len(x))]
    psi = [(i - median(x)) / omhatx for i in x]
    i1 = len([i for i in psi if i < -1])
    i2 = len([i for i in psi if i > 1])
    sx = [0 if i < -1 else i for i in x]
    sx = [0 if i > 1 else i for i in sx]
    pbos = (sum(sx) + omhatx * (i2 - i1)) / (len(x) - i1 - i2)
    return pbos