def robmed_print(x, *args):
    print("Call:")
    print(x['call'])
    print("\nMediated effect:", round(x['ab.est'], 4))
    print("Confidence interval:", round(x['CI.ab'], 4))
    print("p-value:", round(x['p.value'], 4))
    print("\n")