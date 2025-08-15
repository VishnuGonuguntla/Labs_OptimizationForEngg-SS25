# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# p0 = np.array([[180],[0]], dtype=float)
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[[0], [0]], [[1], [2]]], dtype=float)
# fk = np.array([[2], [3]], dtype=float)
# myErrorVector = leastSquaresModel(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[1], [1]]

import numpy as np
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = ""
    return matrnr


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0: # check for positive alpha0
        raise TypeError('range of alpha0 is wrong!')

    if beta <= 1: # check for sufficiently large beta
        raise TypeError('range of beta is wrong!')

    if verbose: # print information
        print('Start levenbergMarquardtDescent...') # print start

    countIter = 0 # counter for loop iterations

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    p = p0.copy()                                                                   # assign value of p as a copy of column vector p0
    alphak = alpha0                                                                 # assign value of alphak as  alpha0
    J = R.jacobian(p)                                                               # Assign J as Jacobian of p using error Vector class
    r = R.residual(p)                                                               # Assign r as Residual of p using error Vector class
    while np.linalg.norm(J.T @ r) > eps:                                            # Tolerance check - norm of scalar product of Jacobian and residual should be lesser than epsilon.
        
        dk = PCG.PrecCGSolver(J.T@J + alphak*np.eye(J.shape[1]),-J.T @ r,1e-5,0)    # Attain descent direction (dk) using Preconditioned CG Solver class.
        if R.residual(p + dk).T@R.residual(p + dk) < R.residual(p).T@R.residual(p): # Condition check - Scalar Product of Residual at next point should be lesser than the current point 
            p = p + dk                                                              # Update value of p with dk 
            alphak = alpha0                                                         # reset value of alphak with alpha0
        else:                                                                       # Else condition
            alphak = beta*alphak                                                    # update value of alphak by multiplying it with beta.
        J = R.jacobian(p)                                                           # Assign J as Jacobian of p using error Vector class (Again for updation in the loop)
        r = R.residual(p)                                                           # Assign J as Residual of p using error Vector class (Again for updation in the loop)
        countIter += 1                                                              # Iteration count
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        gradp = R.jacobian(p).T @ R.residual(p) # store final gradient
        print('levenbergMarquardtDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradp)) # print termination and gradient information

    return p