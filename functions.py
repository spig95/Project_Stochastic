from parameters import *

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

def u(X) :
    ''' Define the velocity field '''
    pre = q / (2*np.pi * (X[0]**2 + X[1]**2)) 
    return np.array([u1+pre*X[0],u2 + pre*X[1]])

def CI(mean, std, N, confidence):
    ''' Compute the confidence interval at the desired confidence level 
    '''
    alfa = 1 - confidence
    C_alfa2 = st.t.ppf(1-alfa/2,N-1)
    lowerB = mean - C_alfa2*std/np.sqrt(N)
    upperB = mean + C_alfa2*std/np.sqrt(N)
    return lowerB, upperB

#############################################################
############## POINT A   ####################################
#############################################################


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


def BasicMonteCarlo(X0, walks, N, T = 1, confidence = 0.95, seed = 1, tol = 1e-6,
                    PDEProb = -1, verbose = 2):
    polluted = np.zeros(walks)

    start = time.time()
    for w in range(walks):
        if (verbose == 2 and w%100 == 0):
            print('Current walk: ', w )
        _, currentTime = NaiveRandomWalk(X0, N, T)
        if currentTime < T - tol:
                polluted[w] = 1
    end = time.time()

    mean = polluted.mean()
    std = np.std(polluted, ddof = 1)
    LB, UB = CI(mean, std, walks, confidence)
    if verbose >=1:
        print(f'\nNumber of simulations: %d. Time needed = %.2f s' % (walks, end-start))
        print(f'Estimated variance: {std}' % (std))
        print(f'The estimated probability at {X0} is: {mean} (using MC)')
        print(f'Confidence interval: [ {mean} +- {UB-mean} ]\twith P = {confidence}%')
        if PDEProb != -1:
            print(f'\nPDE result at {X0} is:  {PDEProb}')
    return mean, std, LB, UB


#############################################################
############## POINT B   ####################################
#############################################################


def deltaTBound(X):
    ''' Computes the bound of deltaT if we want to have an expected step 
        magnitude such that with one step we don't skip the whole well
        X: current position
    '''
    r = np.sqrt(X[0]**2 + X[1]**2) #Current distance from the origin
    Dmax = r #Maximum distance in expectation
    U = u(X) #Velocity field at this position

    delta = sigma**4 + (U[0]**2 + U[1]**2) * (R-r)**2
    bound = (-sigma**2 + np.sqrt(delta) )/(U[0]**2 + U[1]**2)
    return bound


def RandomWalkAdaptiveTimeStep(X0, T = 1 , mindt = 0, maxdt = 10, coeff=1, MAXITERS = 1e4):
    ''' X0: initial position
        T: time limit
        Generate random walk using adaptive time step given by the function deltaTBound.
        The coefficient allows to be more/less conservative wrt the given bound on dt.
        The given dt will be clipped at a maximum value given by the thresholds min/max dt
    '''
    X = []
    X.append(X0)
    
    finalT = 0
    dt = 0 #initialize to 0
    
    # control on the iterations
    iters = 0
    
    while(finalT <= T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt) * norm.rvs(size=2)
        X.append(X0)
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
        
        dt = np.clip( coeff * deltaTBound(X0), mindt, maxdt)
        finalT = finalT + dt

        # if we are inside the well
        if(r<R):
            return np.asarray(X), finalT
        
        # if we are stuck in this walk
        iters = iters + 1
        if(iters > MAXITERS):
            print ('MAXITERS limit reached')
            return np.asarray(X), 0.11111
            
    # if we have "walked" for at time greater than T
    return np.asarray(X), finalT


def AdaptiveTimeStepMonteCarlo(X0, walks, T = 1, confidence = 0.95, seed = 1,
                               mindt = 0, maxdt = 10, coeff = 1, tol = 1e-6, 
                               PDEProb = -1, verbose = 2):
    
    np.random.seed(seed)
    polluted = np.zeros(walks)


    start = time.time()
    for w in range(walks):
        if (verbose == 2 and w%100 == 0):
            print('Current walk: ', w )
        _, currentTime = RandomWalkAdaptiveTimeStep(X0, T = T, mindt = mindt, 
                                                maxdt = maxdt, coeff = coeff)
        if currentTime < T - tol:
                polluted[w] = 1
    end = time.time()

    mean = polluted.mean()
    std = np.std(polluted, ddof = 1)
    LB, UB = CI(mean, std, walks, confidence)
    if verbose >= 1:
        print(f'\nNumber of simulations: %d. Time needed = %.2f s' % (walks, end-start))
        print(f'Estimated variance: {std}')
        print(f'The estimated probability at {X0} is: {mean} (using MC)')
        print(f'Confidence interval: [ {mean} +- {UB-mean} ]\twith P = {confidence}%')
        if PDEProb != -1:
            print(f'\nPDE result at {X0} is:  {PDEProb}')

    return mean, std, LB, UB


#############################################################
############## POINT D   ####################################
#############################################################

def StageWalk(X0, R_in, R_f, T_in, dt):
    finalT = T_in
    r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    while(r > R_f and finalT < T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt)* norm.rvs(size=2)
        finalT = finalT + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    return X0, finalT

def StageWalk_Plot(X0, R_in, R_f, T_in, dt):
    finalT = T_in
    r = np.sqrt( X0[0]**2 + X0[1]**2 )
    X = [X0]
    while(r > R_f and finalT < T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt)* norm.rvs(size=2)
        X.append(X0)
        finalT = finalT + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    return np.asarray(X), finalT

def f1212(X0, T0, Y, n, R, stage, root):    
    # Y is an array that contains the number of hits that has been generated from the path
    
    if stage == 0:
        for starting_path in range(int(n[stage])):
            X, finalT = StageWalk(X0, R[stage], R[stage+1], T0, dt)
            if(finalT < T):
                Y = f1212(X, finalT, Y, n, R, stage + 1, starting_path)
            
    elif stage != n.shape[0]-1:
        for i in range(int(n[stage])):
            X, finalT = StageWalk(X0, R[stage], R[stage+1], T0, dt)
            if(finalT < T):
                Y = f1212(X, finalT, Y, n, R, stage + 1, root)
            
    else:
        for i in range(int(n[stage])):
            X, finalT = StageWalk(X0, R[stage], R[stage+1], T0, dt)
            if(finalT < T):
                Y[root] = Y[root] + 1
                #don't call the function again, we are in the final layer
            
    return Y 

