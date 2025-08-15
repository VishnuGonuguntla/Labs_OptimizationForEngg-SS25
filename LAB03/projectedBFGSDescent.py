# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the reduced BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = ""
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start projectedBFGSDescent...') # print start

    countIter = 0 # counter for number of loop iterations
    xp = P.project(x0) # initialize with projected starting point

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    n = x0.shape[0]                                                                                      # Assign value of n as size of x0
    Hk = np.eye(n)                                                                                       # Assign Hk as Identity matrix of order n
    Ak = P.activeIndexSet(xp)                                                                            # Assign Ak with active Indices of projection of  x0 i.e., (xp)
    while np.linalg.norm(xp - P.project(xp - f.gradient(xp))) > eps:                                     # Condition check: Magnitude of difference of xp and projection of difference of xp and gradient of xp should be lesser than the tolerance.
        dk = PCG.PrecCGSolver(Hk,-f.gradient(xp),1e-6,0)                                                 # Attain direction vector using PreCG Solver
        if f.gradient(xp).T@dk > 0:                                                                      # Descent Direction check
            dk = -f.gradient(xp)                                                                         # Assign descent direction as negative of gradient of xp
            Hk = np.eye(n)                                                                               # Assign Hk as Identity marix of order n (Again)
        tk = PB.projectedBacktrackingSearch(f,P,xp,dk,1e-3,1e-2,0)                                       # Attain stepsize (t) using Projected Backtracking Search Algorithm and assign it to tk.
        xpl = P.project(xp + tk*dk)                                                                      # Assign xpl (x+) with projection of (xp + tk*dk)
        Apl = P.activeIndexSet(xpl)                                                                      # Assign Apl with Active Indices of (A+)
        if not np.array_equal(Apl, Ak):                                                                  # Condition check comparing arrays values of Apl and Ak.
            for i in Apl:                                                                                # for loop over Apl (A+)
                Hk[:,i] = 0                                                                              # Assign all columns with index i as 0.
                Hk[i,:] = 0                                                                              # Assign all rows with index i as 0.
                Hk[i,i] = 1                                                                              # Assign all diagonal indices with 1.
        else:                                                                                            # Else condition
            delgk = f.gradient(P.project(xp + tk*dk)) - f.gradient(xp)                                   # assigning delgk as difference of gradients of projection of gradient of the new point and the projection of gradient of xp
            delxk = xpl - xp                                                                             # assigning delxk as difference of xpl (x+) and projection of x0.
            if delgk.T@delxk <= eps**2:                                                                  # condition check scalar product of delgk and delxk
                Hk = np.eye(n)                                                                           # Assign Hk as identity marix of order n
            else:                                                                                        # Else Condtion
                Hk = Hk + (delgk@delgk.T)/(delgk.T@delxk) - (Hk@delxk@(Hk@delxk).T)/(delxk.T@(Hk@delxk)) # Update Hk with formula given Lemma 6.6 in Page:82 of the OptEngScript.pdf
                for i in Apl:                                                                            # for loop over Apl (A+)
                    Hk[:,i] = 0                                                                          # Assign all columns with index i as 0.
                    Hk[i,:] = 0                                                                          # Assign all rows with index i as 0.
                    Hk[i,i] = 1                                                                          # Assign all diagonal indices with 1.
        xp = xpl                                                                                         # Assign xp as xpl.
        Ak = Apl                                                                                         # Assign Ak as Apl.
        countIter +=1                                                                                    # Increase value of countIter (iteration count)1
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        gradx = f.gradient(xp) # get gradient
        stationarity = np.linalg.norm(xp - P.project(xp - gradx)) # get stationarity
        print('projectedBFGSDescent terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity)) # print termination

    return xp
# Test cases:
# p = np.array([[1], [1]])
# myObjective = SO.simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = PB.projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 0)
# print(xmin)
# should return xmin close to [[1],[1]]