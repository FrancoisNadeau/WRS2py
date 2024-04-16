def print_spp(x, *args):
    print("Call:")
    print(x['call'])
    print("\nTest statistics:")
    print(round(x['test'], 4))
    print("\nTest whether the corresponding population parameters are the same:")
    print("p-value:", round(x['p.value'], 5))
    print("\n")