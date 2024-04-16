import numpy as np

def outogk(x, sigmamu=taulc, v=gkcov, op=True, SEED=False,
           beta=max([.95, min([.99, 1/np.shape(x)[0] + .94])]),
           n_iter=1, plotit=True, *args, **kwargs):
    if not isinstance(x, np.ndarray):
        raise ValueError("x should be a matrix")
    x = elimna(x)
    if not op:
        temp = ogk.pairwise(x, sigmamu=sigmamu, v=v, beta=beta, n_iter=n_iter, *args, **kwargs)
        vals = hard.rejection(temp['distances'], p=np.shape(x)[1], beta=beta, *args, **kwargs)
        flag = (vals == 1)
        vals = np.arange(1, np.shape(x)[0]+1)
        outid = vals[~flag]
        keep = vals[flag]
        if np.shape(x)[1] == 2 and plotit:
            plt.plot(x[:,0], x[:,1], 'o', label='X')
            plt.plot(x[flag,0], x[flag,1], 'o', label='Flag')
            if np.sum(~flag) > 0:
                plt.plot(x[~flag,0], x[~flag,1], 'o', label='Keep')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
    if op:
        temp = out(x, cov.fun=ogk, beta=beta, plotit=plotit, SEED=SEED)
        outid = temp['out.id']
        keep = temp['keep']
    return {'out.id': outid, 'keep': keep, 'distances': temp['dis']}