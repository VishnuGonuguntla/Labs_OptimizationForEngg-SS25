# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = ""
    return matrnr


def SUCSGradient(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Start SUCSGradient...') # print start

    grad_f_h = x.copy() # initialize simplex gradient of f

    # INCOMPLETE CODE STARTS
    n = x.shape[0]                                                # Assign n as size of x
    for i in range(n):                                            # For Loop over the range of size of Array x
        ej = np.zeros((n,1)); ej[i,0] = 1                         # Initialize ej as Column vector with magnitude 0 and assign the loop index as 1. 
        xp = x + h*ej                                             # Assign xp by adding the product of h and ej to x
        xm = x - h*ej                                             # Assign xm by subtracting product of h and ej from x
        grad_f_h[i,0] = (f.objective(xp) - f.objective(xm))/(2*h) # Calculate gradient of f_h by dividing the difference of  objective of xp and objective of xm by twice the scale value (h).
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('SUCSGradient terminated with gradient =', grad_f_h) # print termination

    return grad_f_h


def SUCSStencilFailure(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Check for SUCSStencilFailure...') # print start of check

    stenFail = 1 # initialize stencil failure with true

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    n = x.shape[0]                                                               # Assign n as size of x
    for i in range(n):                                                           # For Loop over range of size of Array x
        ej = np.zeros((n,1)); ej[i,0] = 1                                        # Initialize ej as Column vector with magnitude 0 and assign the loop index as 1.
        xp = x + h*ej                                                            # Assign xp by adding the product of h and ej to x
        xm = x - h*ej                                                            # Assign xm by subtracting product of h and ej from x
        if f.objective(xp) < f.objective(x) or f.objective(xm) < f.objective(x): # Stencil Failure condition from the Algorithm
            stenFail = 0                                                         # Assign stenFail value to 0 for return of function SUCSStencilFailure 
            break                                                                # break the for loop
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        print('SUCSStencilFailure check returns ', stenFail) # print termination

    return stenFail
