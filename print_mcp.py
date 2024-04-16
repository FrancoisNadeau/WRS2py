def print_mcp(x, *args):
    print("Call:\n")
    print(x['call'])
    print("\n")
    if len(x['effects'][0][0]) > 1:
        facA = np.column_stack((x['effects'][0][0], x['effects'][0][1], x['effects'][0][2]))
    else:
        facA = [x['effects'][0][0], x['effects'][0][1], x['effects'][0][2]]
    if len(x['effects'][1][0]) > 1:
        facB = np.column_stack((x['effects'][1][0], x['effects'][1][1], x['effects'][1][2]))
    else:
        facB = [x['effects'][1][0], x['effects'][1][1], x['effects'][1][2]]
    if len(x['effects'][2][0]) > 1:
        facAB = np.column_stack((x['effects'][2][0], x['effects'][2][1], x['effects'][2][2]))
    else:
        facAB = [x['effects'][2][0], x['effects'][2][1], x['effects'][2][2]]
    df = np.vstack((facA, facB, facAB))
    df = pd.DataFrame(df)
    df.index = x['contrasts'].columns
    df.columns.values[3] = "p-value"
    df.columns.values[0] = "psihat"
    print(df.round(5))
    if 'x$alpha.crit' in locals():
        print("\nThe critical alpha level is ", x['alpha.crit'], ".", sep = "")
    print("\n")