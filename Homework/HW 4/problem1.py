import numpy as np
import scipy
import plotly.express as px

x = np.linspace(0,2,10000)
y = 35 * scipy.special.erf(x / (2 * np.sqrt((0.138 * 10 ** -6) * 60*60*24*60))) - 15
print(y)

# Plot!
fig = px.line(x = x, y = y)
fig.update_layout(
    title="Root Finding for Freezing Ground Temperature",
    xaxis_title="Distance below surface [meters]",
    yaxis_title="f(x)",
        yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e')
)
# fig.show()

# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]




# use routines    
f = lambda x: 35 * scipy.special.erf(x / (2 * np.sqrt((0.138 * 10 ** -6) * 60*60*24*60))) - 15
a = 0
b = 2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

tol = 1e-13

[astar,ier, count] = bisection(f,a,b,tol)
print('the approximate root is',astar)
print('the error message reads:',ier)
print('f(astar) =', f(astar))
print('after ' + str(count) + ' iterations')           


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
f1 = lambda x: x - (35 * scipy.special.erf(x / (2 * np.sqrt((0.138 * 10 ** -6) * 60*60*24*60))) - 15) / (35 / (2 * np.sqrt((0.138 * 10 ** -6) * 60*60*24*60)) * (2/np.sqrt(np.pi)) * np.exp(-(x / (2 * np.sqrt((0.138 * 10 ** -6) * 60*60*24*60)))**2))

Nmax = 100
tol = 1e-13
# test f1 '''
x0 = 0.01
[x, xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point for x0 = 0.01 is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print('All x values are: ', x)

Nmax = 100
tol = 1e-13
# test f1 '''
x0 = 2
[x, xstar,ier] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point for x0 = 2 is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print('All x values are: ', x)