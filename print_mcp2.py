def print_mcp2(x, *args):
    print("Call:")
    print(x['call'])
    print()
    df = x['comp']
    gnames1 = x['fnames'][x['comp'][:, 0]]
    gnames2 = x['fnames'][x['comp'][:, 1]]
    df.index = [f"{gnames1[i]} vs. {gnames2[i]}" for i in range(len(gnames1))]
    df.columns = ['Group 1', 'Group 2'] + list(df.columns[2:])
    if df.columns[6] == "p.crit":
        sig = x['comp'][:, 5] < x['comp'][:, 6]
        df = df.round(5)
        df['sig'] = sig
        print(df.iloc[:, 2:])
    else:
        print(df.iloc[:, 2:].round(5))
    print()