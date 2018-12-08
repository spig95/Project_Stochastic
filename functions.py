from parameters import *

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt

def u(X) :
    ''' Define the velocity field '''
    pre = q / (2*np.pi * (X[0]**2 + X[1]**2)) 
    return np.array([u1+pre*X[0],u2 + pre*X[1]])

def NaiveRandomWalk(X0, N, T):
    ''' X0: initial position
        N: number of steps
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
        if(r<R):
            break
    
    return np.asarray(X), finalT

def deltaTBound(X):
    ''' Computes the bound of deltaT if we want to have an expected step 
        magnitude such that with one step we don't skip the whole well
        X: current position
    '''
    r = np.sqrt(X[0]**2 + X[1]**2) #Current distance from the origin
    Dmax = r - R #Maximum distance in expectation
    U = u(X) #Velocity field at this position

    delta = sigma**4 + (U[0]**2 + U[1]**2) * (R-r)**2
    bound = (-sigma**2 + np.sqrt(delta) )/(U[0]**2 + U[1]**2)
    return bound

def RandomWalkAdaptiveTimeStep(X0, T):
    ''' X0: initial position
        T: time limit
    '''
    X = []
    X.append(X0)
    
    finalT = 0
    dt = 0 #initialize to 0
    coeff = 0.7  # If we want to be more/less conservative wrt the bound
    
    # control on the iterations
    MAXITERS = 1e4
    iters = 0
    
    while(finalT <= T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt) * norm.rvs(size=2)
        X.append(X0)
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
        
        dt = deltaTBound(X0)
        finalT = finalT + dt

        # if we are inside the well
        if(r<R):
            return np.asarray(X), finalT
        
        # if we are stuck in this walk
        iters = iters + 1
        if(iters > MAXITERS):
            print ('MAXITERS limit reached')
            return np.asarray(X), 111
            
    # if we have "walked" for at time greater than T
    return np.asarray(X), finalT


def CI(mean, std, N, confidence):
    ''' Compute the confidence interval at the desired confidence level 
    '''
    alfa = 1 - confidence
    C_alfa2 = st.t.ppf(1-alfa/2,N-1)
    lowerB = mean - C_alfa2*std/np.sqrt(N)
    upperB = mean + C_alfa2*std/np.sqrt(N)
    return lowerB, upperB
