def print_pbcor(x, *args):
    print("Call:")
    print(x['call'])
    print("\nRobust correlation coefficient:", round(x['cor'], 4))
    print("Test statistic:", round(x['test'], 4))
    print("p-value:", round(x['p.value'], 5))
    if not math.isnan(x['cor_ci'][0]):
        print("\nBootstrap CI: [", round(x['cor_ci'][0], 4), "; ", round(x['cor_ci'][1], 4), "]\n\n", sep = "")