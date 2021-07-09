import numpy as np
from sklearn.metrics import mean_squared_error
import AR
import SIS

def rmse(SIS_bool, T, N, L, Q, R):
    rmse = np.zeros((T,2))

    # Generating AR process
    a_true, a_clean = AR.process(N,Q)

    for i in range(T):
        S, ex = AR.AR_observation(a_true, R, N)
        if(SIS_bool):
            a, w = SIS.SIS(N, L, S, Q, R)
        else:
            a, w = SIS.SIR(N, L, S, Q, R)
        
        a_av, a_var = SIS.estimate(N, a, w, L)
        rmse[i,0] = mean_squared_error(a_clean[:,0], a_av[:,0])**0.5
        rmse[i,1] = mean_squared_error(a_clean[:,1], a_av[:,1])**0.5
        
    return [rmse[:,0].mean(), rmse[:,0].std()], [rmse[:,1].mean(), rmse[:,1].std()]
    
def neff(w):
    return 1. / np.sum(np.square(w))