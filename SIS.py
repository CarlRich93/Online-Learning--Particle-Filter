import numpy as np
import scipy.stats as sp
from filterpy.monte_carlo import systematic_resample
import Stats

def state(a, n, N, alpha=0.1):
    A0 = np.array([1.2, -0.4])
    a[:,0] = A0[0]*np.ones(len(a)) + alpha*np.cos(2*np.pi*n/N)*np.ones(len(a)) 
    a[:,1] = A0[1]*np.ones(len(a)) + alpha*np.sin(np.pi*n/N)*np.ones(len(a))
    return a
    
def observation(L, a, x):
    s_hat = np.zeros(L)
    for l in range(L):
        s_hat[l] = np.dot(a[l].reshape(1,2), x.reshape(2,1))
    return s_hat
    
def init(L,Q,N):
    a0 = np.zeros((L,2))
    a_next = state(a0, 0.0, N)
    a0[:,0] = np.random.normal(a_next[:,0], Q, size=L) #initialise a set of randomly distributed particles
    a0[:,1] = np.random.normal(a_next[:,1], Q, size=L)
    w0 = np.array([1/L]*L) # initialise weights
    return a0, w0
    
def sample(L, a_minus, n, N, Q):
    a_next = state(a_minus, n, N)
    a_particles = np.zeros((L,2))
    a_particles[:,0] = np.random.normal(a_next[:,0], Q)
    a_particles[:,1] = np.random.normal(a_next[:,1], Q)
    return a_particles
    
def evaluate(L, a, x, s, w, R):
    s_hat = observation(L, a, x)
    for l in range(L):
        w[l] *= sp.norm.pdf(s, loc=s_hat[l], scale=R) + 1.e-300
    return w
  
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))
    
    return weights
      
def estimate(N, a, w, L):
    a_av = np.zeros((N,2))
    a_var = np.zeros((N,2))
    for n in range(N):
        a_av[n,0] = np.average(a[n,:,0], weights=w[n])
        a_av[n,1] = np.average(a[n,:,1], weights=w[n])
        a_var[n,0] = (1/L)*np.sum((a[n,:,0]-a_av[n,0]*np.ones(L))**2)
        a_var[n,1] = (1/L)*np.sum((a[n,:,1]-a_av[n,1]*np.ones(L))**2)
        
    return a_av, a_var
    
def SIS(N, L, S, Q, R):
    a_evol = np.zeros((N,L,2)) # evolution of particles
    w_evol = np.zeros((N,L)) # evolution of weights
    a_particles, w = init(L,Q,N) # initial particles and weights
    a_evol[0] = a_particles
    w_evol[0] = w
    a_particles = sample(L, a_particles, 1, N, Q)
    a_evol[1] = a_particles
    w_evol[1] = w

    for n in range(2,N):
        a_particles = sample(L, a_particles, n, N, Q)
        a_evol[n] = a_particles # append
        w = evaluate(L, a_particles, S[n-2:n], S[n], w, R)
        w = w/sum(w) #normalise
        w_evol[n] = w 
        
    return a_evol, w_evol
    
def SIR(N, L, S, Q, R):
    a_evol = np.zeros((N,L,2)) # evolution of particles
    w_evol = np.zeros((N,L)) # evolution of weights
    a_particles, w = init(L,Q,N) # initial particles and weights
    a_evol[0] = a_particles
    w_evol[0] = w
    a_particles = sample(L, a_particles, 1, N, Q)
    a_evol[1] = a_particles
    w_evol[1] = w

    for n in range(2,N):
        a_particles = sample(L, a_particles, n, N, Q)
        a_evol[n] = a_particles # append
        w = evaluate(L, a_particles, S[n-2:n], S[n], w, R)
        w = w/sum(w) #normalise
        if (Stats.neff(w) < 0.75*L):
            indexes = systematic_resample(w)
            w = resample_from_index(a_particles, w, indexes)
        w_evol[n] = w 
        
    return a_evol, w_evol