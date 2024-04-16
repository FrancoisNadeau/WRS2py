def dmean(m, tr=0.2, dop=1, cop=2):
    if isinstance(m, list):
        m = matl(m)
    if not isinstance(m, np.ndarray):
        raise ValueError("Data must be stored in a matrix or in list mode.")
    if m.shape[1] == 1:
        if tr == 0.5:
            val = np.median(m)
        elif tr > 0.5:
            raise ValueError("Amount of trimming must be at most 0.5")
        elif tr < 0.5:
            val = np.mean(m, tr)
    if m.shape[1] > 1:
        temp = np.empty(m.shape[0])
        if m.shape[1] != 2:
            if dop == 1:
                temp = fdepth(m, plotit=False, cop=cop)
            elif dop == 2:
                temp = fdepthv2(m)
        if m.shape[1] == 2:
            for i in range(m.shape[0]):
                temp[i] = depth(m[i, 0], m[i, 1], m)
        mdep = np.max(temp)
        flag = (temp == mdep)
        if tr == 0.5:
            if np.sum(flag) == 1:
                val = m[flag, :]
            if np.sum(flag) > 1:
                val = np.apply_along_axis(np.mean, 1, m[flag, :])
        if tr < 0.5:
            flag2 = (temp >= tr)
            if np.sum(flag2) == 0 and np.sum(flag) > 1:
                val = np.apply_along_axis(np.mean, 1, np.asmatrix(m[flag, :]))
            if np.sum(flag2) == 0 and np.sum(flag) == 1:
                val = m[flag, :]
            if np.sum(flag2) == 1:
                val = m[flag2, :]
            if np.sum(flag2) > 1:
                val = np.apply_along_axis(np.mean, 1, m[flag2, :])
    return val