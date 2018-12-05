import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def u(X) :
    pre = q / (2*np.pi * (X[0]**2 + X[1]**2)) 
    return np.array([1+pre*X[0], pre*X[1]])


def RandomWalkAdaptiveTimeStep(X0, T):
    ''' X0: initial position
        T: Final time'''
    X = []
    dt = T/N
    sigmaSqrtDt = sigma * np.sqrt(dt)
    X.append(X0)
    finalT = dt
    for i in range(N-1):
        X0 = X0 + u(X0) * dt + sigmaSqrtDt* norm.rvs(size=2)
        X.append(X0)
        finalT = finalT + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
        if(r<1 | finalT >= 1):
            break
    
    return np.asarray(X), finalT

def deltaTBound(X, sigma, R):
    r = np.sqrt(X[0]**2 + X[1]**2)
    U = u(X)
    delta = sigma**4 + (U[0]**2 + U[1]**2) * (R-r)**2
    bound = - sigma**2 + np.sqrt(delta)
    return bound/(U[0]**2 + U[1]**2)