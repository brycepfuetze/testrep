import numpy as np
import scipy
import plotly.express as px
from numpy.linalg import norm 
from numpy.linalg import inv 

def systemIteration(f1, f2, weightMat, x0, tol, Nmax):

    for its in range(Nmax):
       f1Eval = f1(x0[0], x0[1])
       f2Eval = f2(x0[0], x0[1])
       fEval = np.array([f1Eval,f2Eval])

       x1 = x0 - np.matmul(weightMat, fEval)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

def evalF(x): 
    F = np.zeros(2)
    F[0] = 3* x[0] ** 2 - x[1]**2 
    F[1] = 3 * x[0] * (x[1]**2) - x[0]**3 - 1
    return F
    
def evalJ(x): 
    J = np.array([[6 * x[0], -2 * x[1]], 
        [3 * x[1] ** 2 - 3 * x[1] ** 2, 6 * x[0] * x[1]]])
    J = np.squeeze(J, axis=2) 
    return J

def Newton(x0,tol,Nmax):

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F).reshape((2,1))
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

f1 = lambda x, y: 3*(x**2) - y**2
f2 = lambda x, y: 3 * x * y**2 - x**3  - 1
x0 = 1
y0 = 1
ic = np.array([[x0],[y0]])
weightMat = np.array([[1/6, 1/8],[0, 1/6]])
tol = 1e-10
Nmax = 100

[xstar,ier,its] = systemIteration(f1, f2, weightMat, ic, tol, Nmax)
print('x* = [', xstar[0], ', ', xstar[1], ']')
print('The error message reads:',ier)
print('Number of iterations is:',its)

[xstar,ier,its] =  Newton(ic,tol,Nmax)
print('Newton: x* = [', xstar[0], ', ', xstar[1], ']')
print('Newton: The error message reads:',ier)
print('Newton: Number of iterations is:',its)