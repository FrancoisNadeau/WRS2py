def print_ancova(x):
    print("Call:")
    print(x['call'])
    print()
    if 'se' in x.columns:
        df = pd.DataFrame({
            'n1': x['n1'],
            'n2': x['n2'],
            'diff': x['trDiff'],
            'se': x['se'],
            'lower CI': x['ci.low'],
            'upper CI': x['ci.hi'],
            'statistic': x['test'],
            'p-value': x['p.vals']
        })
        df.index = [f"{x['cnames'][3]} = {x['evalpts']}"]
        df.columns = ['n1', 'n2', 'diff', 'se', 'lower CI', 'upper CI', 'statistic', 'p-value']
    else:
        df = pd.DataFrame({
            'n1': x['n1'],
            'n2': x['n2'],
            'diff': x['trDiff'],
            'lower CI': x['ci.low'],
            'upper CI': x['ci.hi'],
            'statistic': x['test'],
            'p-value': x['p.vals']
        })
        df.index = [f"{x['cnames'][3]} = {x['evalpts']}"]
        df.columns = ['n1', 'n2', 'diff', 'lower CI', 'upper CI', 'statistic', 'p-value']
    print(round(df, 4))
    print()