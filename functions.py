from parameters import *

import numpy as np
import scipy.stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=3) #when we print arrays

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

################################################################################
########################### POINT A ############################################
################################################################################

def NaiveRandomWalk(X0, N, T):
    ''' Simulate a random walk starting from X0. The simulation terminates if we
    are inside the well or if T has elapsed.

    #ARGUMENTS:
    X0: initial position
    N: number of steps
    T: maximum time of the walk
    
    #OUTPUT:
    X: np array with the positions of the walk
    currentT: final time of the walk, less than T if the well is contaminated
    '''
    X = []
    dt = T/N
    sigmaSqrtDt = sigma * np.sqrt(dt)
    X.append(X0)
    currentT = dt
    for _ in range(N-1):
        X0 = X0 + u(X0) * dt + sigmaSqrtDt* norm.rvs(size=2)
        X.append(X0)
        currentT = currentT + dt
        r = np.sqrt( X0[0]**2 + X0[1]**2 )
        # Check if we are inside the well
        if(r<R):
            break
    
    return np.asarray(X), currentT


def BasicMonteCarlo(X0, walks, N, T = 1, confidence = 0.95, tol = 1e-6,
                    PDEProb = -1, seed = 1, verbose = 1):
    '''Naive MC method.

    #ARGUMENTS: 
    X0: initial position
    walks: number of walks to simulate
    N: number of steps per walk
    T: final time
    confidence: confidence interval with this confidence level will be computed
    tol: we consider that the walk didn't go inside the well if the time at the 
         Nth step is smaller than T-tol. This prevents errors due to numerical
         approximations
    PDEProb: PDE solution value at X0. -1 if not available

    #OUTPUT:
    mean: estimated prob
    std: estimated variance
    LB, UB: confidence interval 
    '''
    # Seed
    np.random.seed(seed)
    # Initialize to 0 the outcomes of the walks
    polluted = np.zeros(walks)

    start = time.time()
    for w in range(walks):
        if (verbose == 2 and w % 100 == 0):
            print('Current walk: ', w )
        _, currentTime = NaiveRandomWalk(X0, N, T)
        # Update the w-th outcome
        if currentTime < T - tol:
                polluted[w] = 1
    end = time.time()

    mean = polluted.mean()
    std = np.std(polluted, ddof = 1)
    LB, UB = CI(mean, std, walks, confidence)
    if verbose >=1:
        print(f'\nNumber of simulations: %d. Time needed = %.2f s' 
                % (walks, end-start))
        print(f'Estimated variance: {std}' % (std))
        print(f'The estimated probability at {X0} is: {mean} (using MC)')
        print(f'Confidence interval: [ {mean} +- {UB-mean} ]\twith'
               ' P = {confidence}%')
        if PDEProb != -1:
            print(f'\nPDE result at {X0} is:  {PDEProb}')
    return mean, std, LB, UB


def MCWithPilotRun(X0, walks_pilot, N, precision, T = 1, confidence = 0.95, 
                    seed = 1, tol = 1e-6, PDEProb = -1, verbose = 2):
    '''Same as the naive MC, but exploiting a pilot run with "walks_pilot" walks
    to estimate the variance and choose a number of walks to obtain the desired
    "precision".'''

    # Pilot run with "walks_pilot" walks
    _, std, _, _ = BasicMonteCarlo(X0 = X0, walks = walks_pilot, N = N, T = T, 
                    confidence = confidence, tol = tol, PDEProb = PDEProb,
                    seed = seed, verbose = (verbose >=2)*2 )

    # Compute the necessary walks to obtain a CI smaller than "precision"
    alfa = 1 - confidence
    C_alfa2 = st.norm.ppf(1-alfa/2) # a lot of walks, norm is good approx. of t
    walks = int( (2*C_alfa2*std/precision)**2 )
    if verbose>=2: print('\n')
    if verbose>=1: 
        print('Pilot run terminated.')
        print(f'Calling MC method with {walks} walks...\n')

    # MC simulation with the computed number of walks
    start = time.time()
    mean, std, LB, UB = BasicMonteCarlo(X0 = X0, walks = walks, N = N, T = T, 
                        confidence = confidence, tol = tol, PDEProb = PDEProb, 
                        seed = 2*seed, verbose = (verbose >=2)*2 )
    end = time.time()
    
    # Results
    if verbose == 1: #if it is >1 it has already been printed
        print(f'\nNumber of simulations: %d. Time needed = %.2f s' 
              % (walks, end-start))
        print(f'Estimated variance: {std}' % (std))
        print(f'The estimated probability at {X0} is: {mean} (using MC)')
        print(f'Confidence interval: [ {mean} +- {UB-mean} ]\twith'
               ' P = {confidence}%')
        if PDEProb != -1:
            print(f'\nPDE result at {X0} is:  {PDEProb}')

    return mean, std, LB, UB


################################################################################
############################# POINT B ##########################################
################################################################################


def deltaTBoundOrder1(X):
    ''' Computes the bound of deltaT if we want to have an expected step 
        magnitude such that with one step we don't skip the whole well
        X: current position
    '''
    r = np.sqrt(X[0]**2 + X[1]**2) #Current distance from the origin
    
    return (R-r)**2 / (2* sigma**2)

def deltaTBoundOrder2(X):
    ''' Computes the bound of deltaT if we want to have an expected step 
        magnitude such that with one step we don't skip the whole well
        X: current position
    '''
    r = np.sqrt(X[0]**2 + X[1]**2) #Current distance from the origin

    U = u(X) #Velocity field at this position

    delta = sigma**4 + (U[0]**2 + U[1]**2 + 4 * sigma**4) * (R-r)**2
    bound = (-sigma**2 + np.sqrt(delta) )/(U[0]**2 + U[1]**2 + 4*sigma**4)
    return bound



def RandomWalkAdaptiveTimeStep(X0, deltaT = deltaTBoundOrder1, T = 1, 
                                       mindt = 0.0001, maxdt = 0.05, coeff=0.5):
    ''' Generate random walk using adaptive time step given by the function deltaTBound.
        The coefficient allows to be more/less conservative wrt the given bound on dt.
        The given dt will be clipped at a maximum value given by the thresholds min/max dt
        
        X0: initial position
        T: time limit
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
        
        dt = np.clip( coeff * deltaT(X0), mindt, maxdt)
        finalT = finalT + dt

        # if we are inside the well
        if(r<R):
            return np.asarray(X), True
            
    # If we have "walked" for at time greater than T
    return np.asarray(X), False


def AdaptiveTimeStepMonteCarlo(X0, walks, deltaT = deltaTBoundOrder1, T = 1, confidence = 0.95, seed = 1,
                               mindt = 0.0001, maxdt = 0.05, coeff=0.5, 
                               PDEProb = -1, verbose = 1):
    
    np.random.seed(seed)
    polluted = np.zeros(walks)


    start = time.time()
    for w in range(walks):
        if (verbose == 2 and w%100 == 0):
            print('Current walk: ', w )
        _, isIn = RandomWalkAdaptiveTimeStep(X0, deltaT = deltaT, T = T, mindt = mindt, 
                                                maxdt = maxdt, coeff = coeff)
        if isIn: polluted[w] = 1
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



################################################################################
############################# POINT C ##########################################
################################################################################


def createVectorN(N0, NL, L):
    """
    N0 : number of iteration for the level 0
    NL : number of iteration for the level L, level of interest
    L : number of Levels
    output : N vector of size L+1 with the Nl distributed acording to MLMC paper
    """
    #dtl = dt0 * M**l and N = T / dt
    #N0/Nl = M**l
    M = (N0/NL) ** (1/L)
    #Nl = N0/(M**l)
    return np.array([ round(N0 / (M**l)) for l in range(L+1)],dtype=int) 



def MultilevelFunctionForLDifferentTimesSteps(X_0,N,T,L):
    ''' 
    Function that computes walk with different timesteps
    X_0: initial position
    N: vector of the number of steps
    T: Final time
    L : Level to which we are interested walks on L elements of X
    
    return : areIn, the vector of boolean values, areIn[l] = True means the walk on level l has reached the well'''
    
    dt = T/N

    sigmaSqrtDt = sigma * np.sqrt(dt)

    finalT = dt
    areIn = np.full(len(N), False)

    X = np.outer(X_0,np.ones(L+1)).T


    for i in range(N[L]-1):
        Norm = norm.rvs(size=2)
        
        for l in range(L+1):
            if ((not areIn[l]) and (finalT[l] <= T)) :
                X[l] = X[l] + u(X[l]) * dt[l] + sigmaSqrtDt[l]* Norm
                r = np.sqrt( X[l,0]**2 + X[l,1]**2 ) 
                if (r < R) : areIn[l] = True
            
        finalT = finalT + dt
        if(areIn.sum() == L+1):
            break
    
    return areIn


def MultilevelFunctionForLDifferentPositions(X_0,N,T,L):
    ''' 
    Function that computes walk with different starting positions
    X_0: initial positions
    N: number of steps
    T: Final time
    L : Level to which we are interested walks on L elements of X
    
    return : areIn, the vector of boolean values, areIn[l] = True means the walk on level l has reached the well
    '''
    dt = T/N
    sigmaSqrtDt = sigma * np.sqrt(dt)
    finalT = dt
    areIn = np.full(np.shape(X_0)[0], False)
    X = np.array(X_0)

    for i in range(N-1):
        Norm = norm.rvs(size=2)
        
        for l in range(L+1):
            if (not areIn[l]) :
                X[l] = X[l] + u(X[l]) * dt + sigmaSqrtDt* Norm
                r = np.sqrt( X[l,0]**2 + X[l,1]**2 ) 
                if (r < R) : areIn[l] = True
            
        finalT = finalT + dt
        if(areIn.sum() == L+1):
            break
    
    return areIn


def MultiLevelMonteCarlo(L, X0, Walks, Functions, N, T = 1, confidence = 0.95, seed = 1, tol = 1e-6,
                    PDEProb = -1, verbose = 1):
    """
    Runs a Multilevel Montecarlo, of L levels, starting at X0
    Input:
        L : is the number of levels, where L will be the level we want to estimate with 0 being the level with smallest variance
        Walks : is a vector of size L+1, where Walks[l] is the number of walks for the level l
        X0 is a L+1 by 2 matrix storing the starting points, X[l,:] being the starting point of the l-th level
        Functions : the function to apply for the walks, returns a l size vector
    Output: 
        E : vector where E[l] is the expectation of the l-ieth level
        VAR : vector where VAR[l] is the variance of the estimator of the l-ieth level, 
        Var : is the variance of the MLMC estimator, 
        Prob : is the estimator/probability computed by the MLMC  , 
        VarNaive : is the variance that would be achieved for a naive walk
    """
    start = time.time()
    
    #E stores the expectation of each level E[l] = E[Pl-P(l-1)] and E[0] = E[P0]
    E = np.zeros(L+1)
    VAR = np.zeros(L+1)
    
    #stores the polluted values, L lines
    polluted = np.empty((L+1,Walks[0]))
    Walks.append(0)
    for l in range(L,0,-1):
        if(verbose == 2) : print('Calculating level', l)

        for w in range(Walks[l+1],Walks[l]):
            areInR = Functions(X0, N, T, l)
            polluted[:,w] = areInR 
        
        E[l] = np.mean(polluted[l,0:Walks[l]] - polluted[l-1,0:Walks[l]])
        VAR[l] = np.std((polluted[l,0:Walks[l]] - polluted[l-1,0:Walks[l]]), ddof = 1)
    
    #runs the P0 walk
    if(verbose == 2) : print('Calculating level 0')
    for w in range(Walks[1],Walks[0]):
        areInR = Functions(X0, N, T,0)
        polluted[:,w] = areInR 
        
    E[0] = np.mean(polluted[0,:])
    VAR[0] = np.std(polluted[0,:], ddof=1)
    Prob = np.sum(E)
    Var = np.sum(VAR/Walks[:-1])
    VarNaive = np.std(polluted[L,0:Walks[L]], ddof=1)/Walks[L]
    
    end = time.time()
    
    if verbose >=1:
        print(f'\nNumber of simulations: %d. Time needed = %.2f s' % 
                (np.sum(Walks), end-start))
        print(f'The estimated probability at {X0} is: {Prob} (using MC)')
        print('with the variance : ', Var)
        print('Whithout the variance reduction ', VarNaive)
        if PDEProb != -1:
            print(f'\nPDE result at {X0} is:  {PDEProb}')
            
    return E, VAR, Var, Prob, VarNaive



################################################################################
############################# POINT D ##########################################
################################################################################

def StageWalk(X0, R_in, R_f, T_in, dt, T = 1):
    ''' 
    Random walk in a stage driven by the velocity field u, starting from X0. 
    
    #ARGUMENTS:
    X0: starting point
    R_in: X0 belongs to a circle of radius R_in
    R_f: ends if we go a circle of radius R_f (i.e in the next stage)
    T_in: starting time of the walk
    dt: timestep
    T: final time, another stopping criteria for the walk

    #OUTPUT:
    X0: final position of the walk
    currentTime: final time reached by this walk. Greater than T if we have not
        reached the next stage R_f
    '''
    # Initialize time and radius
    currentTime = T_in
    r = np.sqrt( X0[0]**2 + X0[1]**2 )
    
    # Check the stopping criteria and eventually update the position
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



def SplittingMethod(X0, T0, dt, Ns, Rs, Y, H, stage, root, T = 1, verbose = 1,
                                                                  seed = 1):    
    '''Implement the splitting method. Recursive function.
    
    #ARGUMENTS:
    X0: starting point
    T0: initial time
    Ns: array of integers. The i-th element represents the number of walks
        that are generated in case of hitting of the i-th stage.
    Rs: radiuses that defines the stages
    Y: array of length Ns[0]. The i-th element of this array counts the times 
        that the offspring of walks generated by the i-th walk at stage 0  
        reaches the well. See sect. 2.4.3 of ISBN 90-365-14320 for more details.
    H: this array contains one element per each stage. The i-th element counts
        the number of hittings of the walks generated on the i-th stage. Namely
        the walks that starts from a point in the circle of radius Rs[i] and 
        reach Rs[i+1] before T.
    stage: the function is recursive. This integer indicate the actual stage.
        When the function is called this should be 0.
    root: by root, we mean which one of the starting walks (the walks generated 
        at the stage 0) has triggered the current call of the function. This 
        allow us to update accordingly the array Y. At the very first call of 
        the function, root has no meaning whatsoever.
    '''    

    ############################################################################
    # First stage, first call of the function.
    if stage == 0:
        np.random.seed(seed)

        # To be sure, we initialize H and Y to 0 and we perform some dimensional
        # checks.
        if verbose >=1:
            print('Checking dimensionality of H, Ns and Y.')
        if(H.shape[0] != Rs.shape[0] -1):
            print('ERROR: H has wrong dimensions')
            return np.nan, np.nan
        if(Ns.shape[0] != Rs.shape[0] -1):
            print('ERROR: Ns has wrong dimensions')
            return np.nan, np.nan
        if(Y.shape[0] != Ns[0]):
            print('ERROR: Y has wrong dimensions')
            return np.nan, np.nan
        # H and Y must be 0 at the beginning
        H = H*0
        Y = Y*0

        if verbose >= 1:
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
            if verbose >=2: print('Root: ', starting_root)
            X, currentTime = StageWalk(X0, Rs[stage], Rs[stage+1], T0, dt)

            # If we hit the next stage before T, we update the hitting counter 
            # of the current stage H[stage] and we call the function again, with
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

def SplittingMethodBalancedGrowth(X0, dt, Rs, Ns, T = 1, multiplier = 2, 
                                                         verbose = 1, seed = 1):
    '''Runs a pilot run of the splitting method to obtain the values Ns and then
    calls the splitting method accordingly.
    '''
    if verbose >= 1:
        print('Splitting method with balanced growth.\n')
        print(f'Radiuses: {Rs}')
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

        print('\nPilot run with walks per stage: ', Ns)
        print('Pilot run starts...')
    
    # Pilot run
    H  = np.zeros(Rs.shape[0]-1)
    Y  = np.zeros(int(Ns[0]))
    _, H = SplittingMethod(X0, 0, dt, Ns, Rs, Y, H, 
                            stage = 0, root = np.nan, T = T, verbose = 0, 
                            seed = seed)
    if H is np.nan:
        return np.nan, np.nan, np.nan

    if verbose >=1:
        print('Pilot run terminated.')

    # Check if the pilot run has been successful
    if (H == 0).any():
        print('\nERROR: in the pilot run some stages have not been hit.')
        print('H = ', H)
        return np.nan, np.nan, np.nan

    # Compute pi and overwerite Ns as 1/pi
    p = np.empty(H.shape[0])
    p[0] = H[0] / Ns[0]
    for i in range(1, H.shape[0]):
        p[i] = H[i] / (H[i-1] * Ns[i])
    Ns = (np.ceil(1/p)).astype(int)
    Ns = Ns*3
    if (verbose >=1):
        print(f'Pilot run results: \n\tH = {H}\n\tp_i = {p} \n\tN = {Ns}')

    if multiplier !=1:
        if verbose >=1 :
            Ns = Ns*multiplier
            print(f'\nChanging the values multiplying by {multiplier}.')
            print(f'New N = {Ns}.')
    
    if verbose >=1: print('\nCalling the splitting method.')
    
    
    # Reinitialize H and Y and call the splitting method
    H  = np.zeros(Rs.shape[0]-1)
    Y  = np.zeros(int(Ns[0]))

    np.random.seed(5*seed) #scipy is based on the numpy seed
    Y, H = SplittingMethod(X0, 0, dt, Ns, Rs, Y, H, 
                            stage = 0, root = np.nan, T = T, 
                            verbose = (verbose >=2) * 2)
    return Y, H, Ns
    

def ComputeEstimatesSplittingMethod(Y, Ns, PDEProb = -1, verbose = 1):
    ''' Compute mean, std and Confidence interval given the results of the 
    splitting method '''
    # Convert to avoid overflows
    Y = Y.astype(np.float64)
    Ns = Ns.astype(np.float64)

    # Formula 2.12 in ISBN 90-365-14320
    mean = Y.sum()/Ns.prod()

    # Formula 2.15 in ISBN 90-365-14320
    std = np.sqrt( np.std(Y, ddof = 1) / ( Ns[0] * ((Ns[1:]**2).prod()) ) )

    # 95% confidence interval
    # NOTE: N here is always high, hence we can take the quantile of the normal
    # distribution 
    C_alfa2 = st.norm.ppf(0.975)
    LB = mean - C_alfa2*std
    UB = mean + C_alfa2*std
    
    if verbose >= 1:
        print(f'Estimated variance: {std}')
        print(f'The estimated probability is: {mean} (using MC)')
        print(f'Confidence interval: [ {mean} +- {UB-mean} ]\twith P = 95%')
        if PDEProb != -1:
            print(f'\nPDE result is:  {PDEProb}')
    
    return mean, std