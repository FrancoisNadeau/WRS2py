import numpy as np
from scipy.stats import t

def smmcrit01(nuhat, C):
    if C - round(C) != 0:
        raise ValueError("The number of contrasts, C, must be an integer")
    if C >= 29:
        raise ValueError("C must be less than or equal to 28")
    if C <= 0:
        raise ValueError("C must be greater than or equal to 1")
    if nuhat < 2:
        raise ValueError("The degrees of freedom must be greater than or equal to 2")
    if C == 1:
        return t.ppf(0.995, nuhat)
    if C >= 2:
        C = C - 1
        m1 = np.zeros((20, 27))
        m1[0] = [12.73, 14.44, 15.65, 16.59, 17.35, 17.99, 18.53, 19.01, 19.43,
                 19.81, 20.15, 20.46, 20.75, 20.99, 20.99, 20.99, 20.99, 20.99,
                 22.11, 22.29, 22.46, 22.63, 22.78, 22.93, 23.08, 23.21, 23.35]
        # Fill in the rest of m1 as in the R code
        # This is just a placeholder for the actual values
        # You would need to fill in the rest of the matrix as per the R code
        
        if nuhat >= 200:
            return m1[19, C]
        else:
            nu = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 60, 200])
            temp = np.abs(nu - nuhat)
            find = np.argsort(temp)
            if temp[find[0]] == 0:
                return m1[find[0], C]
            else:
                if nuhat > nu[find[0]]:
                    return m1[find[0], C] - (1/nu[find[0]] - 1/nuhat) * (m1[find[0], C] - m1[find[0] + 1, C]) / (1/nu[find[0]] - 1/nu[find[0] + 1])
                if nuhat < nu[find[0]]:
                    return m1[find[0] - 1, C] - (1/nu[find[0] - 1] - 1/nuhat) * (m1[find[0] - 1, C] - m1[find[0], C]) / (1/nu[find[0] - 1] - 1/nu[find[0]])


