def print_t1way(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistic: F =", round(x['test'], 4))
    print("Degrees of freedom 1:", round(x['df1'], 2))
    print("Degrees of freedom 2:", round(x['df2'], 2))
    print("p-value:", round(x['p.value'], 5))
    print("\n")
    if 'effsize' in x:
        print("Explanatory measure of effect size:", round(x['effsize'], 2))
        print("Bootstrap CI: [", round(x['effsize_ci'][0], 2), "; ", round(x['effsize_ci'][1], 2), "]\n\n", sep = "")