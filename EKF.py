import numpy as np

def state(theta_post):
    return theta_post # expected next state
    
def observation(theta, x):
    return float( 1/(1 + np.exp(-np.dot(x,theta))) )
    
def linearise(theta, x):
        return np.exp(-np.dot(x,theta))*(observation(theta, x)**2)*x.reshape(1,2)
        
def predict(theta_post, cov_post, sigma_w):
    theta_prior = state(theta_post)
    cov_prior = sigma_w + cov_post
    return theta_prior, cov_prior
    
def update(theta_prior, cov_prior, x, y, R):
    H = linearise(theta_prior, x)
    S = float(np.linalg.inv(H @ cov_prior @ H.T + R))
    K = S*(cov_prior @ H.T) # kalman gain
    theta_post = theta_prior + K*(y - observation(theta_prior, x))
    cov_post = cov_prior - K @ H @ cov_prior
    return theta_post, cov_post
    
def EKF(x, y, R, sigma_w, theta_init, cov_init, N):
    
    theta_evol = np.zeros((N,2,1))
    cov_evol = np.zeros((N,2,2))
    
    # initialise system
    theta = theta_init # post mean theta @ k=0
    cov = cov_init # post var x (@ k=0)
    theta_evol[0] = theta/np.linalg.norm(theta)
    cov_evol[0] = cov
    
    for n in range(1, N):
        theta, cov = predict(theta, cov, sigma_w) # prior x, prior var
        theta, cov = update(theta, cov, x[n], y[n], R) # post x, post var
        theta_evol[n] = theta/np.linalg.norm(theta)
        cov_evol[n] = cov
    
    return theta_evol, cov_evol
 