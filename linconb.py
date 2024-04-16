import numpy as np

def linconb(x, con=0, tr=0.2, alpha=0.05, nboot=599, pr=True, SEED=True):
    if isinstance(x, pd.DataFrame):
        x = x.values
    
    con = np.array(con)
    
    if isinstance(x, np.ndarray):
        x = x.tolist()
    
    if not isinstance(x, list):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    
    J = len(x)
    for j in range(J):
        xx = x[j]
        x[j] = [val for val in xx if not np.isnan(val)]
    
    Jm = J - 1
    d = (J**2 - J) / 2
    
    if np.sum(con**2) == 0:
        con = np.zeros((J, d))
        id = 0
        for j in range(Jm):
            jp = j + 1
            for k in range(jp, J):
                id += 1
                con[j, id] = 1
                con[k, id] = -1
    
    if con.shape[0] != len(x):
        raise ValueError("The number of groups does not match the number of contrast coefficients.")
    
    bvec = np.zeros((J, 2, nboot))
    
    if SEED:
        np.random.seed(2)
    
    nsam = [len(group) for group in x]
    
    for j in range(J):
        print("Working on group", j)
        xcen = np.array(x[j]) - np.mean(x[j], tr)
        data = np.random.choice(xcen, size=len(x[j])*nboot, replace=True).reshape(nboot, len(x[j]))
        bvec[j, :, :] = np.apply_along_axis(trimparts, 1, data, tr)
    
    m1 = bvec[:, 0, :]
    m2 = bvec[:, 1, :]
    
    boot = np.zeros((con.shape[1], nboot))
    
    for d in range(con.shape[1]):
        top = np.apply_along_axis(trimpartt, 1, m1, con[:, d])
        consq = con[:, d]**2
        bot = np.apply_along_axis(trimpartt, 1, m2, consq)
        boot[d, :] = np.abs(top) / np.sqrt(bot)
    
    testb = np.apply_along_axis(np.max, 1, boot)
    ic = int((1 - alpha) * nboot)
    testb = np.sort(testb)
    
    psihat = np.zeros((con.shape[1], 4))
    test = np.zeros((con.shape[1], 4))
    
    psihat[:, 0] = np.arange(1, con.shape[1] + 1)
    test[:, 0] = np.arange(1, con.shape[1] + 1)
    
    for d in range(con.shape[1]):
        testit = lincon1(x, con[:, d], tr, pr=False)
        test[d, 1] = testit["test"][0, 1]
        pval = np.mean(np.abs(testit["test"][0, 1]) < boot[d, :])
        test[d, 3] = pval
        psihat[d, 2] = testit["psihat"][0, 1] - testb[ic] * testit["test"][0, 3]
        psihat[d, 3] = testit["psihat"][0, 1] + testb[ic] * testit["test"][0, 3]
        psihat[d, 1] = testit["psihat"][0, 1]
        test[d, 2] = testit["test"][0, 3]
    
    return {"n": nsam, "psihat": psihat, "test": test, "crit": testb[ic], "alpha": alpha, "con": con}