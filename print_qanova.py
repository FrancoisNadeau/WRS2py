def print_qanova(x, *args):
    print("Call:")
    print(x['call'])
    
    partable = pd.DataFrame(x['p.value'])
    print("\n")
    print(round(partable, 4))
    print("\n")