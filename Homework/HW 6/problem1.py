import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy

def F(x,y):
    return np.array([x**2 + y**2 - 4, np.exp(x) + y -1])
 
def J(x,y):
    return np.array([[2* x,2 * y],
             [np.exp(x), 1]])

def broydenMethod(x, y, F, B0, tol, Nmax):
    count = 0
 
    F0 = F(x,y)
    B0 = J(x,y)
 
    while np.linalg.norm(F0,2) > tol and count < Nmax:
        
        # solve linear system
        P = scipy.linalg.solve(B0,-1*F0)

        # extract new values
        x = x + P[0]
        y = y + P[1]
        F1 = F(x,y)

        # calc Broyden update (the slower version :( ) 
        dF = F1 - F0
        v = P
        r = dF - np.dot(B0,P)
        u = r / np.dot(P,P)
        B1 = B0 + (np.outer (u, v))

        B0 = B1
        F0 = F1
        
        count = count+1
 
    return x, y, count

def lazyNewton(x, y, F, J0, tol, Nmax):
    count = 0
 
    F0 = F(x,y)
    J0 = J(x,y)
 
    while np.linalg.norm(F0,2) > tol and count < Nmax:
        
        # solve linear system
        P = scipy.linalg.solve(J0,-1*F0)

        # extract new values
        x = x + P[0]
        y = y + P[1]
        F1 = F(x,y)

        F0 = F1
        
        count = count+1
 
    return x, y, count

tol = 1e-10
Nmax = 100

x0 = 1
y0 = 1
x, y, count = broydenMethod(x0, y0, F, J, tol, Nmax)
print('x0 = ', x0, ', y0 = ', y0)
print('Number of iterations: ', count)
print('x = ', x)
print('y = ', y)

x0 = 1
y0 = -1
x, y, count = broydenMethod(x0, y0, F, J, tol, Nmax)
print('x0 = ', x0, ', y0 = ', y0)
print('Number of iterations: ', count)
print('x = ', x)
print('y = ', y)

x0 = 0
y0 = 0
print('x0 = ', x0, ', y0 = ', y0)
# x, y, count = broydenMethod(x0, y0, F, J, tol, Nmax)
print('Matrix is singular')
# print('x = ', x)
# print('y = ', y)


print('Lazy Newton (Chord)')

x0 = 1
y0 = 1
#x, y, count = lazyNewton(x0, y0, F, J, tol, Nmax)
print('x0 = ', x0, ', y0 = ', y0)
#print('Number of iterations: ', count)
#print('x = ', x)
#print('y = ', y)
print('Does not converge!')

x0 = 1
y0 = -1
x, y, count = lazyNewton(x0, y0, F, J, tol, Nmax)
print('x0 = ', x0, ', y0 = ', y0)
print('Number of iterations: ', count)
print('x = ', x)
print('y = ', y)

x0 = 0
y0 = 0
print('x0 = ', x0, ', y0 = ', y0)
#x, y, count = lazyNewton(x0, y0, F, J, tol, Nmax)
#print('Number of iterations: ', count)
#print('x = ', x)
#print('y = ', y)
print('Matrix is singular!')