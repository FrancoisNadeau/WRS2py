def print_AKP(x, *args):
    print("Call:\n")
    print(x['call'])
    print("\nAKP effect size:", round(x['AKPeffect'], 2))
    print("Bootstrap CI: [", round(x['AKPci'][0], 2), "; ", round(x['AKPci'][1], 2), "]\n\n", sep = "")