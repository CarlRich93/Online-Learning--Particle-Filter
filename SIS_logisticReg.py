import numpy as np
import scipy.stats as sp
from filterpy.monte_carlo import systematic_resample
import Stats

def state(theta):
    return theta
    
def observation(L, theta, x):
    y_hat = np.zeros(L)
    for l in range(L):
        y_hat[l] = float(1 / (1 + np.exp(-np.dot(theta[l], x))))
    return y_hat
    
def init(L):
    theta0 = np.zeros((L,2))
    theta0[:,0] = np.random.uniform(low=-1.0, high=1.0, size=L) #initialise a set of randomly distributed particles
    theta0[:,1] = np.random.uniform(low=-1.0, high=1.0, size=L)
    w0 = np.array([1/L]*L) # initialise weights
    return theta0, w0
    
def sample(L, theta_minus, Q):
    theta_next = state(theta_minus)
    theta_particles = np.zeros((L,2))
    theta_particles[:,0] = np.random.normal(theta_next[:,0], Q)
    theta_particles[:,1] = np.random.normal(theta_next[:,1], Q)
    return theta_particles
    
def evaluate(L, a, x, s, w, R):
    s_hat = observation(L, a, x)
    for l in range(L):
        w[l] *= sp.norm.pdf(s, loc=s_hat[l], scale=R) + 1.e-300
    return w
    
def evaluate(L, theta, x, w, y, R):
    y_hat = observation(L, theta, x)
    for l in range(L):
        w[l] *= sp.norm.pdf(y, loc=y_hat[l], scale=R) + 1.e-300
    return w
  
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))
    
    return weights
    
def estimate(N, theta, w, L):
    theta_av = np.zeros((N,2))
    theta_var = np.zeros((N,2))
    for n in range(N):
        theta_av[n,0] = np.average(theta[n,:,0], weights=w[n])
        theta_av[n,1] = np.average(theta[n,:,1], weights=w[n])
        theta_var[n,0] = (1/L)*np.sum((theta[n,:,0]-theta_av[n,0]*np.ones(L))**2)
        theta_var[n,1] = (1/L)*np.sum((theta[n,:,1]-theta_av[n,1]*np.ones(L))**2)
        
    return theta_av, theta_var
    
def SIS(N, L, x, y, Q, R):
    theta_evol = np.zeros((N,L,2)) # evolution of particles
    w_evol = np.zeros((N,L)) # evolution of weights
    theta_particles, w = init(L) # initial particles and weights
    theta_evol[0] = theta_particles/np.linalg.norm(theta_particles, axis=1).reshape(L,1)
    w_evol[0] = w

    for n in range(1,N):
        theta_particles = sample(L, theta_particles, Q)
        theta_evol[n] = theta_particles/np.linalg.norm(theta_particles, axis=1).reshape(L,1) # append
        w = evaluate(L, theta_particles, x[:,n], w, y[n], R)
        w = w/sum(w) #normalise
        w_evol[n] = w 
        
    return theta_evol, w_evol
    
def SIR(N, L, x, y, Q, R):
    theta_evol = np.zeros((N,L,2)) # evolution of particles
    w_evol = np.zeros((N,L)) # evolution of weights
    theta_particles, w = init(L) # initial particles and weights
    theta_evol[0] = theta_particles/np.linalg.norm(theta_particles, axis=1).reshape(L,1)
    w_evol[0] = w

    for n in range(1,N):
        theta_particles = sample(L, theta_particles, Q)
        theta_evol[n] = theta_particles/np.linalg.norm(theta_particles, axis=1).reshape(L,1) # append
        w = evaluate(L, theta_particles, x[:,n], w, y[n], R)
        w = w/sum(w) #normalise
        if (Stats.neff(w) < 0.75*L):
            indexes = systematic_resample(w)
            w = resample_from_index(theta_particles, w, indexes)
        w_evol[n] = w 
        
    return theta_evol, w_evol