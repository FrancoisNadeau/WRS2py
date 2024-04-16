def taulc(x, mu_too=False):
    val = tauvar(x)
    if mu_too:
        val[1] = tauloc(x)
        val[2] = val
    return val