def print_pb2(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistic: ", round(x['test'], 4), ", p-value = ", round(x['p.value'], 5), "\n")
    print("95% confidence interval:")
    print(round(x['conf.int'][0], 4), "  ", round(x['conf.int'][1], 4), "\n")