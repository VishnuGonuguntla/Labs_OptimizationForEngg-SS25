# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d
# and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np
import simpleValleyObjective as SO

def matrnr():
    # set your matriculation number here
    matrnr = ""
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x) # store objective
    gradx = f.gradient(x) # store gradient
    descent = gradx.T @ d # store descent value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start WolfePowellSearch...') # print start

    t = 1 # initial step size guess

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    if gradx.T@d > 0 :                                                                      # Check Descent Direction
        raise Exception ("descent direction check fails")                                   # Raising the Exception regarding the Descent Direction
    t = 1; tm = 0 ; tp = 0                                                                  # Initialization of step values ( t, t_+ & t_+ )
    if bool(f.objective(x + t*d) < f.objective(x) + t*sigma*f.gradient(x).T@d) == 0:        # Check for backtracking
        t = t/2                                                                             # Reduce the step-size by half.
        while bool(f.objective(x + t*d) < f.objective(x) + t*sigma*f.gradient(x).T@d) == 0: # Check the same condition as Line 81.
            t = t/2                                                                         # Reduce the step-size by half till it satisfies equation W1 from the algorithm.
        tm = t; tp = 2*t                                                                    # Update t_- and t_+ values.
    elif bool(f.gradient(x+t*d).T@d >= rho*f.gradient(x).T@d) == 1:                         # Check W2 condition from the Algorithm Step 4.
        return t                                                                            # If it satisfies W2 return the step-size value as is.
    else:                                                                                   # If it does not satisfy the above conditons do front-tracking.
        t = 2*t                                                                             # Increase the step-size by doubling the value.
        while bool(f.objective(x + t*d) < f.objective(x) + t*sigma*f.gradient(x).T@d) == 1: # Check W1 condition again and front-track till it satisfies.
            t = 2*t                                                                         # Increase the step-size by doubling the value.
        tm = t/2; tp = t                                                                    # Update t_- and t_+ values.
    t = tm                                                                                  # Update the t_- value after finishing all conditionals for W1 and W2.
    while bool(f.gradient(x+t*d).T@d >= rho*f.gradient(x).T@d) == 0:                        # Check W2 condition and Refine till it satisfies the condition.
        t = 0.5*(tm+tp)                                                                     # Take average of both the values tm and tp.
        if bool(f.objective(x + t*d) < f.objective(x) + t*sigma*f.gradient(x).T@d) == 1:    # Check W1 condition.
            tm = t                                                                          # Assign t to t_-
        else:                                                                               # Else Condtion
            tp = t                                                                          # Assign t tp t_+
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        xt = x + t * d # store solution point
        fxt = f.objective(xt) # get its objective
        gradxt = f.gradient(xt) # get its gradient
        print('WolfePowellSearch terminated with t=', t) # print terminatin and step size
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent) # print Wolfe-Powell checks

    return t

p = np.array([[0], [1]])
# myObjective = SO.simpleValleyObjective(p)
class Obj:
    def __init__(self, p:np.array):
        self.p = p
    def objective(self, x: np.array):
        return x[0,0]**2 + x[1,0]**2
    def gradient(self, x:np.array):
        return np.array([[2*x[0,0]], [2*x[1,0]]])
    def hessian(self, x:np.array):
        return np.array([[2,0],[0,2]])
myObjective = Obj(p)
x = np.array([[-0.04], [-0.04]])
d = np.array([[1/2], [1/2]])
sigma = 0.25
rho = 0.75
t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16
print(t)