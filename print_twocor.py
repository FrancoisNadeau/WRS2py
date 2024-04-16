def print_twocor(x, *args):
    print("Call:")
    print(x['call'])
    print("\nFirst correlation coefficient:", round(x['r1'], 4))
    print("Second correlation coefficient:", round(x['r2'], 4))
    print("Confidence interval (difference):", round(x['ci'], 4))
    if 'p.value' in x:
        print("p-value:", round(x['p.value'], 5), "\n")
    else:
        print("\n")