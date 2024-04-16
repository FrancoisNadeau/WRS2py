import numpy as np
from scipy.stats import median_absolute_deviation
from sklearn.covariance import MinCovDet, EmpiricalCovariance

def pdis(m, MM=False, cop=3, dop=1, center=None):
    m = elimna(m)
    m = np.matrix(m)
    
    if m.shape[1] == 1:
        if center is None:
            center = np.median(m)
        dis = np.abs(m[:, 0] - center)
        if not MM:
            temp = idealf(dis)
            pdis = dis / (temp['qu'] - temp['ql'])
        if MM:
            pdis = dis / mad(dis)
    
    if m.shape[1] > 1:
        if center is None:
            if cop == 1:
                center = dmean(m, tr=0.5, dop=dop)
            if cop == 2:
                center = MinCovDet().fit(m).location_
            if cop == 3:
                center = np.median(m, axis=0)
            if cop == 4:
                center = EmpiricalCovariance().fit(m).location_
            if cop == 5:
                center = smean(m)
        
        dmat = np.empty((m.shape[0], m.shape[0]))
        dmat[:] = np.nan
        
        for i in range(m.shape[0]):
            B = m[i, :] - center
            dis = np.empty(m.shape[0])
            BB = B**2
            bot = np.sum(BB)
            
            if bot != 0:
                for j in range(m.shape[0]):
                    A = m[j, :] - center
                    temp = np.sum(A * B) * B / bot
                    dis[j] = np.sqrt(np.sum(temp**2))
                
                if not MM:
                    temp = idealf(dis)
                    dmat[:, i] = dis / (temp['qu'] - temp['ql'])
                if MM:
                    dmat[:, i] = dis / mad(dis)
        
        pdis = np.apply_along_axis(np.max, 1, dmat, na_rm=True)
    
    return pdis