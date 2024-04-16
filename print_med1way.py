def print_med1way(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistic F:", round(x['test'], 4))

    if not pd.isna(x['crit.val']):
        print("Critical value:", round(x['crit.val'], 4))
    print("p-value:", round(x['p.value'], 5))
    print("\n")