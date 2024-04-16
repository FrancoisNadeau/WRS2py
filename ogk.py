import numpy as np

def ogk(x, sigmamu, v, n_iter, beta, *args):
    if not isinstance(x, np.ndarray):
        raise ValueError("x should be a matrix")
    x = elimna(x)
    temp = ogk_pairwise(x, sigmamu=sigmamu, v=v, n_iter=n_iter, beta=beta, *args)
    return {'center': temp['wcenter'], 'cov': temp['wcovmat']}