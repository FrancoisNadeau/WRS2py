import math

def mestse(x, bend=1.28, *args):
    n = len(x)
    mestse = math.sqrt(sum((ifmest(x, bend, op=2)**2)) / (n * (n-1)))
    return mestse