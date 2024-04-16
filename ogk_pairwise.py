import numpy as np

def ogk_pairwise(X, n_iter=1, sigmamu=taulc, v=gkcov, beta=0.9, *args, **kwargs):
    data_name = X.__name__
    X = np.array(X)
    n = X.shape[0]
    p = X.shape[1]
    Z = X
    U = np.eye(p)
    A = []
    
    for iter in range(1, n_iter+1):
        d = np.apply_along_axis(sigmamu, 0, Z, *args, **kwargs)
        Z = np.divide(Z, d[:, np.newaxis])
        for i in range(p-1):
            for j in range(i+1, p):
                U[j, i] = U[i, j] = v(Z[:, i], Z[:, j], *args, **kwargs)
        
        E = np.linalg.eig(U)[1]
        
        A.append(d * E)
        
        Z = np.dot(Z, E)
    
    sqrt_gamma = np.apply_along_axis(sigmamu, 0, Z, mu_too=True, *args, **kwargs)
    center = sqrt_gamma[0, :]
    sqrt_gamma = sqrt_gamma[1, :]
    
    Z = np.subtract(Z, center)
    Z = np.divide(Z, sqrt_gamma[:, np.newaxis])
    distances = np.sum(Z**2, axis=1)
    
    covmat = np.diag(sqrt_gamma**2)
    for iter in range(n_iter, 0, -1):
        covmat = np.dot(np.dot(A[iter-1], covmat), A[iter-1].T)
        center = np.dot(A[iter-1], center)
    center = center.flatten()
    
    weights = hard_rejection(distances, p, beta=beta, *args, **kwargs)
    sweights = np.sum(weights)
    
    wcenter = np.sum(np.multiply(X, weights[:, np.newaxis]), axis=0) / sweights
    Z = np.subtract(X, wcenter)
    Z = np.multiply(Z, np.sqrt(weights)[:, np.newaxis])
    wcovmat = np.dot(Z.T, Z) / sweights
    
    return {
        'center': center,
        'covmat': covmat,
        'wcenter': wcenter,
        'wcovmat': wcovmat,
        'distances': distances,
        'sigmamu': sigmamu.__name__,
        'v': v.__name__,
        'data_name': data_name,
        'data': X
    }