import numpy as np

def process(N, Q, alpha=0.1):
    "Time varying parameters"
    A0 = np.array([1.2, -0.4])
    A = np.zeros((N,2))
    A_clean = np.zeros((N,2))
    
    for n in range(N):
        A[n,0] = A0[0] + alpha * np.cos(2*np.pi*n/N) + np.random.normal(0.0, Q)
        A[n,1] = A0[1] + alpha * np.sin(np.pi*n/N) + np.random.normal(0.0, Q)
        
    for n in range(N):
        A_clean[n,0] = A0[0] + alpha * np.cos(2*np.pi*n/N)
        A_clean[n,1] = A0[1] + alpha * np.sin(np.pi*n/N)
        
    return A, A_clean
    
def AR_observation(A, R, N):
    ex = np.random.normal(0.0, R, N)
    S = ex.copy()
    for n in range(2, N):
        x = np.array([S[n-1], S[n-2]])
        S[n] = np.dot(x, A[n,:]) + ex[n]
    return S, ex   