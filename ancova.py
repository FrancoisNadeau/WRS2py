import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.spatial.distance import cdist

def ancova(formula, data, tr=0.2, fr1=1, fr2=1, pts=np.nan, **kwargs):
    alpha = 0.05
    if data is None:
        raise ValueError("Data must be provided")
    
    # Extracting variables from formula
    dependent_var, independent_vars = formula.split('~')
    independent_vars = independent_vars.split('+')
    group_var = independent_vars[0].strip()
    covar_var = independent_vars[1].strip()
    
    # Checking if group variable is categorical
    if not pd.api.types.is_categorical_dtype(data[group_var]):
        raise ValueError("Group variable needs to be provided as factor!")
    
    grnames = data[group_var].cat.categories
    if len(grnames) > 2:
        raise ValueError("Robust ANCOVA implemented for 2 groups only!")
    
    # Splitting data based on group
    data_grouped = data.groupby(group_var)
    yy = [data_grouped.get_group(x)[dependent_var] for x in grnames]
    xx = [data_grouped.get_group(x)[covar_var] for x in grnames]
    
    # Ensure the first group has more or equal elements than the second
    if len(xx[0]) < len(xx[1]):
        xx.reverse()
        yy.reverse()
        grnames = grnames[::-1]
    
    # Removing NA values
    xy = [pd.concat([xx[i], yy[i]], axis=1).dropna() for i in range(2)]
    xx = [xy[i][covar_var] for i in range(2)]
    yy = [xy[i][dependent_var] for i in range(2)]
    
    # Fitting LOWESS
    fitted_values = [lowess(yy[i], xx[i], frac=fr1 if i == 0 else fr2, return_sorted=False) for i in range(2)]
    
    # Preparing for tests
    if np.isnan(pts).all():
        pts = np.linspace(min(min(xx[0]), min(xx[1])), max(max(xx[0]), max(xx[1])), 5)
    
    results = []
    for pt in pts:
        # Finding nearest points
        distances = [cdist([[pt]], xx[i][:, None]).min(axis=1) for i in range(2)]
        close_indices = [np.where(distances[i] <= tr)[0] for i in range(2)]
        g1 = yy[0].iloc[close_indices[0]]
        g2 = yy[1].iloc[close_indices[1]]
        
        # Performing t-test
        t_stat, p_value = ttest_ind(g1, g2, equal_var=False)
        
        # Calculating confidence interval
        df = len(g1) + len(g2) - 2
        crit_val = t.ppf(1 - alpha/2, df)
        mean_diff = g1.mean() - g2.mean()
        pooled_se = np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
        ci_low = mean_diff - crit_val * pooled_se
        ci_high = mean_diff + crit_val * pooled_se
        
        results.append({
            'X': pt,
            'n1': len(g1),
            'n2': len(g2),
            'DIF': mean_diff,
            'TEST': t_stat,
            'se': pooled_se,
            'ci.low': ci_low,
            'ci.hi': ci_high,
            'p.value': p_value,
            'crit.val': crit_val
        })
    
    results_df = pd.DataFrame(results)
    return {
        'evalpts': results_df['X'],
        'n1': results_df['n1'],
        'n2': results_df['n2'],
        'trDiff': results_df['DIF'],
        'se': results_df['se'],
        'ci.low': results_df['ci.low'],
        'ci.hi': results_df['ci.hi'],
        'test': results_df['TEST'],
        'crit.vals': results_df['crit.val'],
        'p.vals': results_df['p.value'],
        'fitted.values': fitted_values,
        'cnames': [dependent_var, group_var, covar_var],
        'faclevels': grnames
    }


