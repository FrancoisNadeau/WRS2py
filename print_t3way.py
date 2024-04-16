def print_t3way(x, *args):
    print("Call:")
    print(x['call'])
    print()
    df = pd.DataFrame({'value': [x['Qa'], x['Qb'], x['Qc'], x['Qab'], x['Qac'], x['Qbc'], x['Qabc']],
                       'p.value': [x['A.p.value'], x['B.p.value'], x['C.p.value'], x['AB.p.value'], x['AC.p.value'], x['BC.p.value'], x['ABC.p.value']]})
    df.index = [x['varnames'][1], x['varnames'][2], x['varnames'][3], 
                x['varnames'][1] + ':' + x['varnames'][2], x['varnames'][1] + ':' + x['varnames'][3], x['varnames'][2] + ':' + x['varnames'][3],
                x['varnames'][1] + ':' + x['varnames'][2] + ':' + x['varnames'][3]]
    print(df)
    print()