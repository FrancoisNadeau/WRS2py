def print_mcp1(x, *args):
    print("Call:\n")
    print(x['call'])
    print("\n")
    df = x['comp']
    gnames1 = x['fnames'][x['comp'][:,0]]
    gnames2 = x['fnames'][x['comp'][:,1]]
    df.index = [f"{gnames1[i]} vs. {gnames2[i]}" for i in range(len(gnames1))]
    if df.shape[1] > 6:
        if df.columns[6] == "crit":
            sig = [True if abs(x['comp'][i,6]) > x['comp'][i,7] else False for i in range(df.shape[0])]
            df = df.round(5)
            df['test'] = sig
            df.columns = list(df.columns[:6]) + ['test']
            print(df.iloc[:,2:])
        else:
            print(df.iloc[:,2:].round(5))
    else:
        print(df.iloc[:,2:].round(5))
    print("\n")