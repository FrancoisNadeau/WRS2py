import numpy as np

def hard_rejection(distances, p, beta=0.9, *args, **kwargs):
    d0 = np.quantile(distances, beta) * np.median(distances) / np.quantile(distances, 0.5)
    weights = np.zeros(len(distances))
    weights[distances <= d0] = 1.0
    return weights