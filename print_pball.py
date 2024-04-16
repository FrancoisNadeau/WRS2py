def print_pball(x, *args):
    print("Call:")
    print(x['call'])
    print("\nRobust correlation matrix:")
    if 'pbcorm' in x and x['pbcorm'] is not None:
        print(round(x['pbcorm'], 4))
    else:
        print(round(x['cor'], 4))
    print("\np-values:")
    print(round(x['p.values'], 5))
    if 'H' in x and x['H'] is not None:
        print("\n\nTest statistic H: ", round(x['H'], 4), ", p-value = ", round(x['H.p.value'], 5), "\n\n", sep = "")