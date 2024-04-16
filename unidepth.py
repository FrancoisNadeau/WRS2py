def unidepth(x, pts=None):
    if not isinstance(x, list):
        raise ValueError("x should be a list")
    if pts is None:
        pts = x
    pup = [sum([1 for i in pts if i <= j]) / len(x) for j in x]
    pdown = [sum([1 for i in pts if i < j]) / len(x) for j in x]
    pdown = [1 - i for i in pdown]
    m = [[pup[i], pdown[i]] for i in range(len(x))]
    dep = [min(i) for i in zip(*m)]
    return dep