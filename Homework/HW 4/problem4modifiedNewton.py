import numpy as np
import scipy
import plotly.express as px
     


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

# test functions
f1 = lambda x: x - 4*(np.exp(3*x) - 27*x**6 + 27*x**4 * np.exp(x) - 9*x**2 * np.exp(2*x)) / (3*np.exp(3*x) - 162*x**5 + 108*x**3 * np.exp(x) + 27*x**4 * np.exp(x) - 18*x * np.exp(2*x) - 18*x**2 * np.exp(2*x))

Nmax = 100
tol = 1e-13
# test f1 '''
x0 = 5
[x, xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point for x0 = 0.01 is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print('All x values are: ', x)