def bi2KMSv2(x, y, nullval=0):
    r1 = sum(x)
    n1 = len(x)
    r2 = sum(y)
    n2 = len(y)
    
    alph = [i/100 for i in range(1, 100)]
    for i in range(1, 100):
        irem = i
        chkit = bi2KMS(r1=r1, n1=n1, r2=r2, n2=n2, x=x, y=x, alpha=alph[i-1])
        if chkit['ci'][0] > nullval or chkit['ci'][1] < nullval:
            break
    
    p_value = irem / 100
    if p_value <= 0.1:
        iup = (irem + 1) / 100
        alph = [i/1000 for i in range(1, int(iup*1000)+1)]
        for i in range(len(alph)):
            p_value = alph[i]
            chkit = bi2KMS(r1=r1, n1=n1, r2=r2, n2=n2, x=x, y=x, alpha=alph[i])
            if chkit['ci'][0] > nullval or chkit['ci'][1] < nullval:
                break
    
    est = bi2KMS(r1=r1, n1=n1, r2=r2, n2=n2, x=x, y=y)
    return {'ci': est['ci'], 'est.p1': est['p1'], 'est.p2': est['p2'], 'p.value': p_value}

def bi2KMS(x, y, alpha=0.05):
    r1 = sum(x)
    n1 = len(x)
    r2 = sum(y)
    n2 = len(y)
    
    N = n1 + n2
    u = 0.5
    Dhat = (r1 + 0.5) / (n1 + 1) - (r2 + 0.5) / (n2 + 1)
    psihat = ((r1 + 0.5) / (n1 + 1) + (r2 + 0.5) / (n2 + 1)) / 2
    nuhat = (1 - 2 * psihat) * (0.5 - n2 / N)
    what = sqrt(2 * u * psihat * (1 - psihat) + nuhat**2)
    se = qnorm(1 - alpha / 2) * sqrt(u / (2 * n1 * n2 / N))
    val1 = max([-1, (u * Dhat + nuhat) / what - se])
    ci = [what * sin(asin(val1)) / u - nuhat / u]
    val2 = min([1, (u * Dhat + nuhat) / what + se])
    ci.append(what * sin(asin(val2)) / u - nuhat / u)
    return {'ci': ci, 'p1': r1 / n1, 'p2': r2 / n2}

