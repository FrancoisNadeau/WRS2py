def print_yuen(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistic: ", round(x['test'], 4), " (df = ", round(x['df'], 2), ")", ", p-value = ", round(x['p.value'], 5), "\n", sep="")
    print("\nTrimmed mean difference: ", round(x['diff'], 5), "\n")
    print("95 percent confidence interval:")
    print(round(x['conf.int'][0], 4), "   ", round(x['conf.int'][1], 4), "\n\n")
    if 'effsize' in x:
        print("Explanatory measure of effect size:", round(x['effsize'], 2), "\n\n")