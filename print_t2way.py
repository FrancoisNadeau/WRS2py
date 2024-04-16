def print_t2way(x, *args):
    print("Call:")
    print(x['call'])
    print()
    if 'med2way' not in x['call']:
        if not pd.isna(x['Qa']):
            df = pd.DataFrame({'value': [x['Qa'], x['Qb'], x['Qab']], 'p.value': [x['A.p.value'], x['B.p.value'], x['AB.p.value']]})
        else:
            df = pd.DataFrame({'p.value': [x['A.p.value'], x['B.p.value'], x['AB.p.value']]})
        df.index = [x['varnames'][1], x['varnames'][2], x['varnames'][1] + ':' + x['varnames'][2]]
        print(df.round(4))
        print()
    else:
        df = pd.DataFrame({'value': [x['Qa'], x['Qb'], x['Qab']], 'p.value': [x['A.p.value'], x['B.p.value'], x['AB.p.value']]})
        df.index = [x['varnames'][1], x['varnames'][2], x['varnames'][1] + ':' + x['varnames'][2]]
        df = df.round(4)
        free = ['F(' + str(x['dim'][0]-1) + ', Inf)', 'F(' + str(x['dim'][1]-1) + ', Inf)', 'Chisq(' + str(np.prod(x['dim']-1)) + ')']
        df1 = pd.DataFrame({'value': df.iloc[:,0], 'df': free, 'p.value': df.iloc[:,1]})
        print(df1)
        print()