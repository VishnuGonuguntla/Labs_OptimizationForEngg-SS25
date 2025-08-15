# Optimization for Engineers - Dr.Johannes Hild
# projected Wolfe-Powell line search

# Purpose: Find t to satisfy f(P(x+t*d))<f(x) + sigma*gradf(x).T@(P(x+t*d)-x) with P(x+t*d)-x being a descent direction
# and in addition but only if x+t*d is inside the feasible set gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition
# and in addition if x+t*d is inside the feasible set, the second Wolfe-Powell condition holds

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# rho = 0.75
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
# should return t = 0.5

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = ""
    return matrnr


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, rho=1.0e-2, verbose=0):
    xp = P.project(x) # initialize with projected starting point
    fx = f.objective(xp) # get current objective
    gradx = f.gradient(xp) # get current gradient
    descent = gradx.T @ d # descent direction check value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start projectedBacktracking...') # print start

    t = 1 # starting guess for t

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    xk = x ; dk = d                                                                                                                                                                                      # Assign values of value and direction to local variables
    if np.array_equal(P.project(xk + dk), xk):                                                                                                                                                           # Condition Check
        raise Exception ('xk is stationary.')                                                                                                                                                            # Stationary Check and raising error
    tp = 0; tm = 0                                                                                                                                                                                       # Initialize values of step size (t+) and (t-) to 0;
    if bool(gradx.T@(P.project(xk + t*dk)-xk) < 0) and bool(f.objective(P.project(xk + t*dk))<= fx + sigma*gradx.T@(P.project(xk + t*dk)-xk))==0:                                                        # W1 Condition for backtracking
        t = t/2                                                                                                                                                                                          # Decreasing stepsize (t) by half.
        while bool(gradx.T@(P.project(xk + t*dk)-xk) < 0) and bool(f.objective(P.project(xk + t*dk))<= fx + sigma*gradx.T@(P.project(xk + t*dk)-xk))==0:                                                 # W1 condition check until it is true
            t = t/2                                                                                                                                                                                      # Decreasing stepsize (t) by half.
        tm = t                                                                                                                                                                                           # negative step size is left as it is.
        tp = 2*t                                                                                                                                                                                         # Positive stepsize is increasing by a factor of 2.
    elif bool(not np.array_equal(xk + t*dk,P.project(xk + t*dk))) or bool(f.gradient(P.project(xk + t*dk)).T@dk >= rho*gradx.T@dk) :                                                                     # W2 Condition Check.
        return t                                                                                                                                                                                         # End the Projected Backtracking Search with the current step size (t)
    else:                                                                                                                                                                                                # Else
        t = 2*t                                                                                                                                                                                          # Increase stepsize by a factor of 2. 
        while bool(gradx.T@(P.project(xk + t*dk)-xk) < 0) and bool(f.objective(P.project(xk + t*dk))<= fx + sigma*gradx.T@(P.project(xk + t*dk)-xk)) and np.array_equal(P.project(xk + t*dk),xk + t*dk): # W1 and check of Projection at next point
            t = 2*t                                                                                                                                                                                      # Increase the stepsize by factor of 2 until it satisfies the above condition.
        tm = t/2                                                                                                                                                                                         # Decrease negative step size by a factor of 2. 
        tp = t                                                                                                                                                                                           # Positive stepsize is increasing as it is.
    t = tm                                                                                                                                                                                               # Assign step-size as negative step-size (t-)
    while (bool(not np.array_equal(xk + t*dk,P.project(xk + t*dk))) or bool(f.gradient(P.project(xk + t*dk)).T@dk >= rho*gradx.T@dk)) == False:                                                          # W2 condition check for refining
        t = (tm + tp)/2                                                                                                                                                                                  # Average the values of (t-) and (t+)
        if bool(gradx.T@(P.project(xk + t*dk)-xk) < 0) and bool(f.objective(P.project(xk + t*dk))<= fx + sigma*gradx.T@(P.project(xk + t*dk)-xk)):                                                       # W1 Condition check
            tm = t                                                                                                                                                                                       # Assign (t-) as stepsize (t)
        else:                                                                                                                                                                                            # Else
            tp = t                                                                                                                                                                                       # Assign (t+) as stepsize (t)
    # INCOMPLETE CODE ENDS

    if verbose: # print verbose information
        xt = P.project(x + t * d) # get x+td for found step size t
        fxt = f.objective(xt) # get objective value at this point
        print('projectedBacktracking terminated with t=', t) # print termination
        print('Sufficient decrease: ', fxt, '<=', fx+t*sigma*descent) # print result of sufficient decrease check

    return t