import numpy as np

def data_gen(N, class_prob, mu, cov):
    """draws samples from 2 multivariate distributions and returns array of all samples, corresponding vector of class labels
        and 2 arrays of class separated data points"""
    rng0 = np.random.default_rng()
    class0 = rng0.multivariate_normal(mu, cov, size=int(class_prob[0]*N)) # generating samples from gaussian 0
    rng1 = np.random.default_rng()
    class1 = rng1.multivariate_normal(-mu, cov, size=int(class_prob[1]*N)) # generating samples from gaussian 1
    data = np.concatenate((class0, class1))
    
    y = []
    for i in range(len(class0)):
        y.append(0)
    for i in range(len(class1)):
        y.append(1)
      
    return data, np.asarray(y), class0, class1

def prob(x, theta):
    "for a given theta (boundary), returns the posterior prob of class given observation"
    prob_c0 = np.zeros(len(x))
    for n in range(len(x)):
        prob_c0[n] = 1/(1 + np.exp(-np.dot(theta,x[n])))
    return prob_c0