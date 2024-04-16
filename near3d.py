import numpy as np
from scipy.spatial import distance

def near3d(x, pt, fr=0.8, m):
    if not isinstance(x, np.ndarray):
        raise ValueError("Data are not stored in a matrix.")
    
    dis = np.sqrt(distance.mahalanobis(x, pt, m['cov']))
    dflag = dis < fr
    
    return dflag