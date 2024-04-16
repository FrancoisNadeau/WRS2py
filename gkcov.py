def gkcov(x, y, gk_sigmamu=taulc, *args):
    val = 0.25 * (gk_sigmamu(x + y, *args) - gk_sigmamu(x - y, *args))
    return val