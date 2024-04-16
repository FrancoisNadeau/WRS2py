import numpy as np

def kron(m1, m2):
    m1 = np.array(m1)
    m2 = np.array(m2)
    kron = []
    for i in range(m1.shape[0]):
        m3 = m1[i, 0] * m2
        for j in range(1, m1.shape[1]):
            m3 = np.column_stack((m3, m1[i, j] * m2))
        if i == 0:
            kron = m3
        else:
            kron = np.vstack((kron, m3))
    return kron