import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

def mcp2atm(formula, data, tr=0.2, **kwargs):
    if data is None:
        mf = ols(formula).model.data
    else:
        mf = ols(formula, data=data).model.data
    
    J = len(mf.iloc[:, 1].unique())
    K = len(mf.iloc[:, 2].unique())
    alpha = 0.05
    grp = np.nan
    op = False
    JK = J * K
    nfac = mf.groupby([mf.columns[1], mf.columns[2]]).size().unstack()
    nfac1 = nfac.loc[mf.iloc[:, 1].unique(), mf.iloc[:, 2].unique()]
    
    data = data.dropna(subset=mf.columns)
    data = data.sort_values(by=[mf.columns[1], mf.columns[2]])
    data['row'] = np.repeat(np.arange(1, nfac1.shape[0] + 1), nfac1.sum(axis=1))
    
    dataMelt = pd.melt(data, id_vars=['row', mf.columns[1], mf.columns[2]], value_vars=mf.columns[0])
    dataWide = dataMelt.pivot_table(index='row', columns=[mf.columns[1], mf.columns[2]], values='value', aggfunc='mean')
    dataWide = dataWide.droplevel(0, axis=1)
    
    x = [mf.iloc[:, 0].values[mf.iloc[:, 1:].apply(tuple, axis=1) == i] for i in np.arange(1, JK + 1)]
    if not np.isnan(grp[0]):
        yy = x
        x = []
        for j in range(len(grp)):
            x.append(yy[int(grp[j]) - 1])
    
    for j in range(JK):
        xx = x[j]
        x[j] = xx[~np.isnan(xx)]
    
    for j in range(JK):
        temp = x[j]
        temp = temp[~np.isnan(temp)]
        x[j] = temp
    
    temp = con2way(J, K)
    conA = temp['conA']
    conB = temp['conB']
    conAB = temp['conAB']
    
    if not op:
        FactorA = lincon1(x, con=conA, tr=tr, alpha=alpha)
        FactorB = lincon1(x, con=conB, tr=tr, alpha=alpha)
        FactorAB = lincon1(x, con=conAB, tr=tr, alpha=alpha)
    
    AllTests = np.nan
    if op:
        FactorA = np.nan
        FactorB = np.nan
        FactorAB = np.nan
        con = np.column_stack((conA, conB, conAB))
        AllTests = lincon1(x, con=con, tr=tr, alpha=alpha)
    
    cnamesA = mf.columns[1]
    dnamesA = [cnamesA + str(i) for i in range(1, conA.shape[1] + 1)]
    cnamesB = mf.columns[2]
    dnamesB = [cnamesB + str(i) for i in range(1, conB.shape[1] + 1)]
    dnamesAB = [':'.join(ss) for ss in np.array(np.meshgrid(dnamesA, dnamesB)).T.reshape(-1, 2)]
    contrasts = pd.DataFrame(np.column_stack((conA, conB, conAB)), columns=dnamesA + dnamesB + dnamesAB)
    contrasts.index = dataWide.columns
    
    outA = {'psihat': FactorA[2][:, 'psihat'], 'conf.int': FactorA[2][:, ['ci.lower', 'ci.upper']], 'p.value': FactorA[2][:, 'p.value']}
    outB = {'psihat': FactorB[2][:, 'psihat'], 'conf.int': FactorB[2][:, ['ci.lower', 'ci.upper']], 'p.value': FactorB[2][:, 'p.value']}
    outAB = {'psihat': FactorAB[2][:, 'psihat'], 'conf.int': FactorAB[2][:, ['ci.lower', 'ci.upper']], 'p.value': FactorAB[2][:, 'p.value']}
    effects = {cnamesA: outA, cnamesB: outB, cnamesA + ':' + cnamesB: outAB}
    
    result = {'effects': effects, 'contrasts': contrasts, 'call': None}
    result['call'] = None  # Replace None with the appropriate call object
    
    return result