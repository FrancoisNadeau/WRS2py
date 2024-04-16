def selby2(m, grpc, coln=None):
    if coln is None:
        raise ValueError("The argument coln is not specified")
    if len(grpc) > 4:
        raise ValueError("The argument grpc must have length less than or equal to 4")
    x = []
    ic = 0
    if len(grpc) == 2:
        cat1 = selby(m, grpc[0], coln)["grpn"]
        cat2 = selby(m, grpc[1], coln)["grpn"]
        for i1 in range(len(cat1)):
            for i2 in range(len(cat2)):
                temp = []
                it = 0
                for i in range(len(m)):
                    if sum(m[i, [grpc[0], grpc[1]]] == [cat1[i1], cat2[i2]]) == 2:
                        it += 1
                        temp.append(m[i, coln])
                if temp:
                    ic += 1
                    x.append(temp)
                    if ic == 1:
                        grpn = np.matrix([cat1[i1], cat2[i2]])
                    else:
                        grpn = np.vstack([grpn, [cat1[i1], cat2[i2]]])
    if len(grpc) == 3:
        cat1 = selby(m, grpc[0], coln)["grpn"]
        cat2 = selby(m, grpc[1], coln)["grpn"]
        cat3 = selby(m, grpc[2], coln)["grpn"]
        for i1 in range(len(cat1)):
            for i2 in range(len(cat2)):
                for i3 in range(len(cat3)):
                    temp = []
                    it = 0
                    for i in range(len(m)):
                        if sum(m[i, [grpc[0], grpc[1], grpc[2]]] == [cat1[i1], cat2[i2], cat3[i3]]) == 3:
                            it += 1
                            temp.append(m[i, coln])
                    if temp:
                        ic += 1
                        x.append(temp)
                        if ic == 1:
                            grpn = np.matrix([cat1[i1], cat2[i2], cat3[i3]])
                        else:
                            grpn = np.vstack([grpn, [cat1[i1], cat2[i2], cat3[i3]]])
    if len(grpc) == 4:
        cat1 = selby(m, grpc[0], coln)["grpn"]
        cat2 = selby(m, grpc[1], coln)["grpn"]
        cat3 = selby(m, grpc[2], coln)["grpn"]
        cat4 = selby(m, grpc[3], coln)["grpn"]
        for i1 in range(len(cat1)):
            for i2 in range(len(cat2)):
                for i3 in range(len(cat3)):
                    for i4 in range(len(cat4)):
                        temp = []
                        it = 0
                        for i in range(len(m)):
                            if sum(m[i, [grpc[0], grpc[1], grpc[2], grpc[3]]] == [cat1[i1], cat2[i2], cat3[i3], cat4[i4]]) == 4:
                                it += 1
                                temp.append(m[i, coln])
                        if temp:
                            ic += 1
                            x.append(temp)
                            if ic == 1:
                                grpn = np.matrix([cat1[i1], cat2[i2], cat3[i3], cat4[i4]])
                            else:
                                grpn = np.vstack([grpn, [cat1[i1], cat2[i2], cat3[i3], cat4[i4]]])
    return {"x": x, "grpn": grpn}