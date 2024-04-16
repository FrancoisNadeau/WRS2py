def print_onesampb(x, *args, **kwargs):
    print("Call:")
    print(x['call'])
    print()
    print("Robust location estimate:", round(x['estimate'], 4))
    cistr = str(1 - x['alpha']) + "% confidence interval:"
    print(cistr, round(x['ci'], 4))
    print("p-value:", round(x['p.value'], 4))
    print()