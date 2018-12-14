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

def StageWalk(X0, R_in, R_f, T_in, dt, T = 1):
    ''' 
    Random walk driven by the velocity field u, starting from X0. X0 belongs to 
    a circle of radius R_in and the walk starts at time T_in.
    The walk ends if we are inside a circle of radius R_f (i.e in the next stage)
    or if the time is greater than T.

    #OUTPUT:
    X0: final position of the walk
    currentTime: final time reached by this walk. Greater than T if we have not
        reached the next stage R_f
    '''
    currentTime = T_in
    r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    while(r > R_f and currentTime < T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt)* norm.rvs(size=2)
        currentTime = currentTime + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    return X0, currentTime

def StageWalk_Plot(X0, R_in, R_f, T_in, dt, T=1):
    '''
    Same as StageWalk, but returns an array representing the walk.
    Used for plots
    '''
    currentTime = T_in
    r = np.sqrt( X0[0]**2 + X0[1]**2 )
    X = [X0]
    while(r > R_f and currentTime < T):
        X0 = X0 + u(X0) * dt + sigma * np.sqrt(dt)* norm.rvs(size=2)
        X.append(X0)
        currentTime = currentTime + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    return np.asarray(X), currentTime



def SplittingMethod(X0, T0, dt, Ns, Rs, Y, H, stage, root, T = 1, verbose = 1):    
    '''Implement the splitting method. Recursive function.
    
    #ARGUMENTS:
    X0: starting point
    T0: initial time
    Ns: array of integers. The i-th element represents the number of walks
        that are generated in case of hitting of the i-th stage.
    Rs: radiuses that defines the stages
    Y: array of length Ns[0]. The i-th element of this array counts the times 
        that the offspring of walks generated by the i-th walk at stage 0 reaches 
        the well. See sect. 2.4.3 of ISBN 90-365-14320 for more details.
    H: this array contains one element per each stage. The i-th element counts
        the number of hittings of the walks generated on the i-th stage. Namely
        the walks that starts from a point in the circle of radius Rs[i] and 
        reach Rs[i+1] before T.
    stage: the function is recursive. This integer indicate the actual stage.
        When the function is called this should be 0.
    root: by root, we mean which one of the starting walks (the walks generated 
        at the stage 0) has triggered the current call of the function. This 
        allow us to update accordingly the array Y. At the very first call of the
        function, root has no meaning whatsoever.
    '''    
    
    ############################################################################
    # First stage, first call of the function.
    if stage == 0:

        # To be sure, we initialize H and Y to 0 and we perform some dimensional
        # checks.
        print('Checking dimensionality of H, Ns and Y.')
        if(H.shape[0] != Rs.shape[0] -1):
            print('H has wrong dimensions')
            return np.nan, np.nan
        if(Ns.shape[0] != Rs.shape[0] -1):
            print('Ns has wrong dimensions')
            return np.nan, np.nan
        if(Y.shape[0] != Ns[0]):
            print('Y has wrong dimensions')
            return np.nan, np.nan
        # H and Y must be 0 at the beginning
        H = H*0
        Y = Y*0

        if verbose == 1:
            print(f'Radiuses: {Rs}')
            print(f'Walks per stage: {Ns}')
            print(f'Timestep: {dt}')
            fig, ax = plt.subplots(figsize = [4,4])
            for radius in Rs[:-1]:
                theta = np.linspace(0,2*np.pi,100)
                ax.plot(radius*np.cos(theta),radius*np.sin(theta),'k-')
            circle = plt.Circle((0, 0), Rs[-1], color='k')
            ax.add_artist(circle)
            ax.plot(X0[0],X0[1],'.', ms = 20, label='X0')
            ax.set_title('Stages configuration.')
            ax.legend()
            plt.show()

            print('\nSplitting method starts...')


         # We generate Ns[0] stage walks and initialize the values of the roots
        for starting_root in range(int(Ns[stage])):
            X, currentTime = StageWalk(X0, Rs[stage], Rs[stage+1], T0, dt)

            # If we hit the next stage before T, we update the hitting counter of
            # the current stage H[stage] and we call the function again, with
            # the correct stage number and the correct root
            if(currentTime < T):
                H[stage] = H[stage] + 1
                Y, H = SplittingMethod(X, currentTime, dt, Ns, Rs, Y, H, 
                                    stage + 1, starting_root)



    ############################################################################
    # Intermediate stage, same as before, but we don't have to change the root
    # when we call the next instance of SplittingMethod, since the roots are 
    # defined only in the firs stage        
    elif stage != Ns.shape[0]-1:
        for i in range(int(Ns[stage])):
            X, currentTime = StageWalk(X0, Rs[stage], Rs[stage+1], T0, dt)
            if(currentTime < T):
                H[stage] = H[stage] + 1
                Y, H = SplittingMethod(X, currentTime, dt, Ns, Rs, Y, H, 
                                    stage + 1, root)
    


    ############################################################################
    # Final stage. We don't call again the SplittingMethod, but we update the 
    # values of Y[root]            
    else:
        for i in range(int(Ns[stage])):
            X, currentTime = StageWalk(X0, Rs[stage], Rs[stage+1], T0, dt)
            if(currentTime < T):
                H[stage] = H[stage] + 1
                Y[root] = Y[root] + 1
            
    return Y, H


def ComputeEstimatesSplittingMethod(Y, Ns, PDEProb = -1, verbose = 1):
    ''' Compute mean, std and Confidence interval given the results of the 
    splitting method '''
    # Formula 2.12 in ISBN 90-365-14320
    mean = Y.sum()/Ns.prod()

    # Formula 2.15 in ISBN 90-365-14320
    std = np.sqrt( np.std(Y, ddof = 1) / ( Ns[0] * ((Ns[1:]**2).prod()) ) )
    
    if verbose == 1:
        print('Splitting method results.')
        print(f'Estimated probability: {mean}')
        print(f'Estimated variance: {std}')
        if PDEProb != -1:
                print(f'\nPDE result is:  {PDEProb}')
    return mean, std