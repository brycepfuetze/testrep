# import libraries
import numpy as np


# define routines
def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    x = np.array(x0)

    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        x = np.append(x, x1)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            return [np.delete(x, -1), xstar,ier]
        x0 = x1
    xstar = x1
    ier = 1
    return [np.delete(x, -1), xstar, ier]

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    print('the order of the equation is')
    print('log(|p_{n+1} -p|) = log(lambda) + alpha*log(|p_n-p|) where:')
    print('lambda = ' + str(np.exp(fit[1])))
    print('alpha = ' + str(fit[0]))
    return [fit, diff1, diff2]


# test functions
f1 = lambda x: 12 / (1+x)
 # Number 1: -16 + 6*x + 12/x
 # Number 2: 2/3 * x + 1/(x**2) 
 # Number 3: 12 / (1+x)

Nmax = 100
tol = 1e-10
# test f1 '''
x0 = 2
[x, xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print('All x values are: ', x)

[fit, diff1, diff2] = compute_order(x, xstar)