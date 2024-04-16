def print_t1waybt(x, *args):
    print("Call:")
    print(x['call'])
    print("\nEffective number of bootstrap samples was", x['nboot.eff'], ".\n")
    print("Test statistic:", round(x['test'], 4))
    print("p-value:", round(x['p.value'], 5))
    print("Variance explained:", round(x['Var.Explained'], 3))
    print("Effect size:", round(x['Effect.Size'], 3))
    print("\n")