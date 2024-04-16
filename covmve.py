import numpy as np
from sklearn.covariance import MinCovDet

def covmve(x):
    val = MinCovDet().fit(x)
    return {'center': val.location_, 'cov': val.covariance_}