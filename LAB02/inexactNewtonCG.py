# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton CG

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dA = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]]

import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA
import noHessianObjective as nHO

def matrnr():
    # set your matriculation number here
    matrnr = "" 
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start inexactNewtonCG...') # print start

    countIter = 0 # counter for number of loop iterations
    xk = x0 # initialize starting iteration

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    xk = np.array(x0, dtype=np.float64)                                                         # Initialize xk value and define it with x0 with np.float64 as datatype
    etak = min(1/2, np.sqrt(np.linalg.norm(f.gradient(xk))))*np.linalg.norm(f.gradient(xk))     # Initialize η_k as defined in the Algorithm 6.3
    while np.linalg.norm(f.gradient(xk)) > eps:                                                 # Stationary Check with a tolerance value.
        xj = xk.copy()                                                                          # Initialize xj as a copy of xk.
        rj = f.gradient(xk)                                                                     # Initialize rj with gradient of f at xk.
        dj = -rj.copy()                                                                         # Initialize dj with negative of gradient of f at xk.
        while np.linalg.norm(rj) > etak:                                                        # Check norm of rj with η_k defined in the Algorithm.
            dA = DHA.directionalHessApprox(f,xk,dj,1e-3,0)                                      # Assign value of dA with Hessian Approximation using the noHessianObjective class.
            rhoj = dj.T@dA                                                                      # Assign rhoj with dot product of dj and dA.
            if rhoj <= eps*np.linalg.norm(dj)**2:                                               # Check if rhoj  is lesser than product of norm of dj and ε.
                # print("Curvature Fail")                                                       # If it fails it causes curvature fail.
                break                                                                           # Have to stop the inner while loop.
            tj = np.linalg.norm(rj)**2/rhoj                                                     # Assign value of tj with the ratio of norm of rj and ρj.
            xj += tj*dj                                                                         # Increase value of xj with element wise vector product of tj and dj.
            rold = rj.copy()                                                                    # Assign variable rold as a copy of rj
            rj = rold + tj*dA                                                                   # Update rj as sum of rold and element wise product of tj and dA.
            beta = np.linalg.norm(rj)**2/np.linalg.norm(rold)**2                                # Assign beta value as a ratio of square of norm of rj and rold.
            dj = -rj + beta*dj                                                                  # Update dj as sum of negative of rj and product of beta and dj.
        if np.array_equal(xj,xk):                                                               # Equality check of xj and xk.
            dk = -f.gradient(xk)                                                                # Assign dk as negative of gradient of f at xk.
        else:                                                                                   # Else condition
            dk = xj-xk                                                                          # Assign dk as difference of xj and xk.
        tk = WP.WolfePowellSearch(f, xk, dk, 1e-3, 1e-2, 0)                                     # Attain step-size tk using WolfePowellSearch Class.
        xk += tk*dk                                                                             # Increase the value of xk with element-wise product of tk and dk.
        etak = min(1/2, np.sqrt(np.linalg.norm(f.gradient(xk))))*np.linalg.norm(f.gradient(xk)) # Calculate etak values with newly obtained xk vector.
        countIter += 1                                                                          # Counter for checking no. of iterations.
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        stationarity = np.linalg.norm(f.gradient(xk)) # store stationarity value
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', stationarity) # print termination with stationarity value

    return xk
