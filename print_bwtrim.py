def print_bwtrim(x, *args, **kwargs):
    print("Call:")
    print(x['call'])
    print()
    dfx = pd.DataFrame({'value': [x['Qa'], x['Qb'], x['Qab']],
                        'df1': [x['A.df'][0], x['B.df'][0], x['AB.df'][0]],
                        'df2': [x['A.df'][1], x['B.df'][1], x['AB.df'][1]],
                        'p.value': [x['A.p.value'], x['B.p.value'], x['AB.p.value']]})
    dfx.index = [x['varnames'][1], x['varnames'][2], x['varnames'][1] + ':' + x['varnames'][2]]
    dfx = dfx.round(4)
    print(dfx)
    print()