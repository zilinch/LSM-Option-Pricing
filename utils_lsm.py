import numpy as np
from numpy.random import default_rng
from sklearn import linear_model

# define basis function for Laguerre polynomials
def BasisFunctLaguerre(X, k):
    # X: n-dimensional vector
    # k: number of basis
    # return A: a nxk matrix

    if k <= 0:
        raise ValueError('k needs to be at least 1')
    if k == 1:
        A = np.column_stack((np.ones_like(X), (1-X)))
    elif k == 2:
        A = np.column_stack((np.ones_like(X), (1-X), 0.5*(2-4*X+X**2)))
    elif k == 3:
        A = np.column_stack((np.ones_like(X), (1-X), 0.5*(2-4*X+X**2), 1/6*(6-18*X+9*X**2-X**3)))
    elif k == 4:
        A = np.column_stack((np.ones_like(X), (1-X), 0.5*(2-4*X+X**2), 1/6*(6-18*X+9*X**2-X**3), 1/24*(24-96*X+72*X**2-16*X**3+X**4)))
    elif k == 5:
        A = np.column_stack((np.ones_like(X), (1-X), 0.5*(2-4*X+X**2), 1/6*(6-18*X+9*X**2-X**3), 1/24*(24-96*X+72*X**2-16*X**3+X**4), 1/120*(120-600*X+600*X**2-200*X**3+25*X**4-X**5)))
    elif k == 6:
        A = np.column_stack((np.ones_like(X), (1-X), 0.5*(2-4*X+X**2), 1/6*(6-18*X+9*X**2-X**3), \
                             1/24*(24-96*X+72*X**2-16*X**3+X**4), 1/120*(120-600*X+600*X**2-200*X**3+25*X**4-X**5),\
                             1/720*(X**6-36*X**5+450*X**4-2400*X**3+5400*X**2-4320*X+720)
                            ))
    else:
        raise ValueError('Too many basis functions requested')
    
    return A


# define basis function for Laguerre polynomials
def BasisFunctHermite(X, k):
    if k <= 0:
        raise ValueError('k needs to be at least 1')
    if k == 1:
        A = np.column_stack((np.ones_like(X), 2*X))
    elif k == 2:
        A = np.column_stack((np.ones_like(X), 2*X, 4*X**2-2))
    elif k == 3:
        A = np.column_stack((np.ones_like(X), 2*X, 4*X**2-2, 8*X**3-12*X))
    elif k == 4:
        A = np.column_stack((np.ones_like(X), 2*X, 4*X**2-2, 8*X**3-12*X, \
                            16*X**4-48*X**2+12
                            ))
    elif k == 5:
        A = np.column_stack((np.ones_like(X), 2*X, 4*X**2-2, 8*X**3-12*X, \
                            16*X**4-48*X**2+12, 32*X**5-160*X**3+120*X
                            ))
    elif k == 6:
        A = np.column_stack((np.ones_like(X), 2*X, 4*X**2-2, 8*X**3-12*X, \
                            16*X**4-48*X**2+12, 32*X**5-160*X**3+120*X, \
                            64*X**6-480*X**4+720*X**2-120
                            ))
    else:
        raise ValueError('Too many basis functions requested')

    return A

def BasisFunct(X, k, polyn):
    if polyn == 'Laguerre':
        return BasisFunctLaguerre(X, k)
    elif polyn == 'Hermite':
        return BasisFunctHermite(X, k)
    else:
        raise ValueError('Polynomial Type not supported')
    


# compute analytical solution for beta in Least Square regression
def computeBeta(L, Y):
    return np.linalg.inv(L.T @ L) @ L.T @ Y


def computeBetaReg(L, Y, reg):
    reg = linear_model.Ridge(alpha=reg)
    reg.fit(L, Y)
    return reg.coef_.T


# LSM Algorithm for PUT option
def LSM_put(T, r, sigma, K, S0, N, M, k, polyn='Laguerre', reg = None, rng = default_rng(42) ):
    dt = T/N
    t = np.linspace(0, T, N+1)
    z = rng.standard_normal((int(M/2), 1)) # Monte Carlo Simulation
    w = (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * np.vstack((z, -z))
    S = S0 * np.exp(w)
    P = np.maximum(K - S, 0)  # Payoff at time T
    exe_bound = dict()
    opt_prices = dict()

    for i in range(N-1, 0, -1):  
        z = rng.standard_normal((int(M/2), 1)) # Monte Carlo Simulation
        w = t[i] * w / t[i+1] + sigma * np.sqrt(dt * t[i] / t[i+1]) * np.vstack((z, -z))
        S = S0 * np.exp(w)
        itmP = np.where(K - S > 0)[0]  # In-the-money Paths Index
        if len(itmP) == 0:
            return np.mean(P * np.exp(-r*dt)), exe_bound, opt_prices
        X = S[itmP]
        Y = P[itmP] * np.exp(-r*dt)
        
        # Perform Regression
        A = BasisFunct(X, k, polyn)
        if reg:
            beta = computeBetaReg(A, Y, reg) #kx1
        else:
            beta = computeBeta(A, Y)
 
        C = A @ beta #nx1
        
        E = K - X  # Value of immediate exercise

        exP = itmP[np.any(C < E, axis=1)] #itmP[C < E] #Paths where it's better to exercise
        
        rest = np.setdiff1d(np.arange(int(M)), exP) #Rest of the Path
        
        P[exP] = E[np.any(C < E, axis=1)] #E[C < E]
        exe_bound[t[i]] = P[exP].ravel().tolist()
        P[rest] = P[rest] * np.exp(-r*dt)
        opt_prices[t[i]] = np.mean(P[rest])

    u = np.mean(P * np.exp(-r*dt))
    return u, exe_bound, opt_prices






# LSM Algorithm for CALL option
def LSM_call(T, r, sigma, K, S0, N, M, k, polyn = 'Laguerre', reg = None, rng = default_rng(42) ):
    dt = T/N
    t = np.linspace(0, T, N+1)
    z = rng.standard_normal((int(M/2), 1)) # Monte Carlo Simulation
    w = (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * np.vstack((z, -z))
    S = S0 * np.exp(w)
    P = np.maximum(S - K, 0)  # Payoff at time T
    exe_bound = dict()
    opt_prices = dict()

    for i in range(N-1, 0, -1):  
        z = rng.standard_normal((int(M/2), 1)) # Monte Carlo Simulation
        w = t[i] * w / t[i+1] + sigma * np.sqrt(dt * t[i] / t[i+1]) * np.vstack((z, -z))
        S = S0 * np.exp(w)
        itmP = np.where(S - K > 0)[0]  # In-the-money Paths Index
        if len(itmP) == 0:
            return np.mean(P * np.exp(-r*dt)), exe_bound, opt_prices
        X = S[itmP]
        Y = P[itmP] * np.exp(-r*dt)
        
        # Perform Regression
        A = BasisFunct(X, k, polyn)
        if reg:
            beta = computeBetaReg(A, Y, reg)
        else:
            beta = computeBeta(A, Y) #kx1
 
        C = A @ beta #nx1
        
        E = X - K  # Value of immediate exercise

        exP = itmP[np.any(C < E, axis=1)] #itmP[C < E] #Paths where it's better to exercise
        
        rest = np.setdiff1d(np.arange(int(M)), exP) #Rest of the Path
        
        P[exP] = E[np.any(C < E, axis=1)] #E[C < E]
        exe_bound[t[i]] = P[exP].ravel().tolist()
        P[rest] = P[rest] * np.exp(-r*dt)
        opt_prices[t[i]] = np.mean(P[rest])

    u = np.mean(P * np.exp(-r*dt))
    return u, exe_bound, opt_prices