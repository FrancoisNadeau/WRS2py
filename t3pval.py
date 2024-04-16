def t3pval(cmat, tmeans, v, h):
    alph = [i/100 for i in range(1, 100)]
    for i in range(99):
        irem = i + 1
        chkit = johan(cmat, tmeans, v, h, alph[i])
        if chkit['teststat'] > chkit['crit']:
            break
    p_value = irem / 100
    if p_value <= 0.1:
        iup = (irem + 1) / 100
        alph = [i/1000 for i in range(1, int(iup * 1000) + 1)]
        for i in range(len(alph)):
            p_value = alph[i]
            chkit = johan(cmat, tmeans, v, h, alph[i])
            if chkit['teststat'] > chkit['crit']:
                break
    if p_value <= 0.001:
        alph = [i/10000 for i in range(1, 11)]
        for i in range(len(alph)):
            p_value = alph[i]
            chkit = johan(cmat, tmeans, v, h, alph[i])
            if chkit['teststat'] > chkit['crit']:
                break
    return p_value