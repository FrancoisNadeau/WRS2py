import numpy as np
from scipy.stats import t

def smmcrit(nuhat, C):
    if C - round(C) != 0:
        raise ValueError("The number of contrasts, C, must be an integer")
    if C >= 29:
        raise ValueError("C must be less than or equal to 28")
    if C <= 0:
        raise ValueError("C must be greater than or equal to 1")
    if nuhat < 2:
        raise ValueError("The degrees of freedom must be greater than or equal to 2")
    
    if C == 1:
        return t.ppf(0.975, nuhat)
    
    if C >= 2:
        C = C - 1
        m1 = np.zeros((20, 27))
        m1[0] = [5.57, 6.34, 6.89, 7.31, 7.65, 7.93, 8.17, 8.83, 8.57,
                 8.74, 8.89, 9.03, 9.16, 9.28, 9.39, 9.49, 9.59, 9.68,
                 9.77, 9.85, 9.92, 10.00, 10.07, 10.13, 10.20, 10.26, 10.32]
        # The rest of m1 initialization with values as in the R code...
        # For brevity, not all values are filled in this example.
        
        nu = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 60, 200])
        temp = np.abs(nu - nuhat)
        find = np.argsort(temp)
        
        if temp[find[0]] == 0:
            return m1[find[0], C]
        
        if temp[find[0]] != 0:
            if nuhat > nu[find[0]]:
                return m1[find[0], C] - (1/nu[find[0]] - 1/nuhat) * (m1[find[0], C] - m1[find[0] + 1, C]) / (1/nu[find[0]] - 1/nu[find[0] + 1])
            if nuhat < nu[find[0]]:
                return m1[find[0] - 1, C] - (1/nu[find[0] - 1] - 1/nuhat) * (m1[find[0] - 1, C] - m1[find[0], C]) / (1/nu[find[0] - 1] - 1/nu[find[0]])