import math

def dnormvar(x):
    return x**2 * math.exp(-x**2/2) / math.sqrt(2*math.pi)