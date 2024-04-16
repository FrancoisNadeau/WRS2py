import pandas as pd
from scipy.stats import chi2_contingency

def disc2_chi_sq(x, y, simulate_p_value=False, B=2000, **kwargs):
    n1 = len(x)
    n2 = len(y)
    g = [1] * n1 + [2] * n2
    d = x + y
    df = pd.DataFrame({'d': d, 'g': g})
    res = chi2_contingency(pd.crosstab(df['d'], df['g']), simulate_p_value=simulate_p_value, B=B)
    return {'X.squared': res[0], 'p.value': res[1]}

disc2com = disc2_chi_sq