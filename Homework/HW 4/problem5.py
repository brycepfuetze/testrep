import numpy as np
import scipy
import plotly.express as px

def secant_method(func, x0, x1, tolerance=1e-7, max_iterations=100):
    x = np.array(x0)

    for iteration in range(max_iterations):
        f_x0 = func(x0)
        f_x1 = func(x1)

        if f_x1 - f_x0 == 0:
            print("Division by zero encountered. No solution found.")
            return None, iteration

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x = np.append(x, x2)

        if abs(x2 - x1) < tolerance:
            return np.delete(x, -1), x2, iteration + 1

        x0, x1 = x1, x2

    print("Maximum iterations reached. No solution found.")
    return None, max_iterations

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
fnewton = lambda x: x - (x**6 - x - 1) / (6*x**5-1)
fsecant = lambda x: x**6 - x - 1

Nmax = 100
tol = 1e-13
# test f1 '''
x0 = 2
[xNewton, xstarNewt,ier] = fixedpt(fnewton,x0,tol,Nmax)
print('the approximate root for Newton is:',xstarNewt)
print('f1(xstar):',fnewton(xstarNewt))
print('Error message reads:',ier)
print('All x values are: ', xNewton)


x0 = 2
x1 = 1
[xSecant, xstarSec, count] = secant_method(fsecant, x0, x1, tol, Nmax)
print('the approximate root for Secant is:',xstarSec)
print('f1(xstar):',fsecant(xstarSec))
print('count:',count)
print('All x values are: ', xSecant)

errSecant = xSecant - xstarSec
errNewton = xNewton - xstarNewt
print("Newton error", errNewton)
print("Secant error", errSecant)

seckplus1 = np.delete(errSecant, 1)
seck = np.delete(errSecant, -1)

newtonkplus1 = np.delete(errNewton, 1)
newtonk = np.delete(errNewton, -1)

fig = px.line(x=newtonk, y=newtonkplus1, log_x=True, log_y=True, title='Error of Newton and Secant Methods', labels={'x1': 'X Axis', 'y1': 'Y Axis'})
fig.add_scatter(x=seck, y=seckplus1, mode='lines', name='Secant Method', line=dict(color='red'))

fig.show()