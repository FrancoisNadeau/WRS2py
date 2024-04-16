def print_robtab(x, *args):
    print("Call:")
    print(x['call'])
    print("\nParameter table: ")
    if x['partable'].shape[1] >= 8:
        print(round(x['partable'], 4))
    else:
        print(x['partable'])
    print("\n")