def print_rmanovab(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistic:", round(x['test'], 4))
    print("Critical value:", round(x['crit'], 4))
    sig = False if x['test'] < x['crit'] else True
    print("Significant:", sig)
    print("\n")