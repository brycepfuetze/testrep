import scipy
import numpy as np

# define routines
def hybrid_method(f, fprime, fdoubleprime,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 
    count = 0
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]


    d = 0.5*(a+b)
    while (abs(f(d) * fprime(d) / fdoubleprime(d))< 1):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
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

      #  x = np.array(x0)

    while (count <Nmax):
        count = count +1
        x1 = x0 - f(x0) / fprime(x0)
        #x = np.append(x, x1)
        if (abs(x1-x0) <tol):
            xstar = x1
            ier = 0
            #return [np.delete(x, -1), xstar,ier]
            return [xstar,ier, count]
        x0 = x1
    xstar = x1
    ier = 1
    return [xstar, ier, count]


# test functions
f = lambda x: np.exp(x**2 + 7*x - 30)
fprime = lambda x: (2*x+7) * np.exp(x**2 + 7*x - 30)
fdoubleprime = lambda x: (2*x+7) * (2*x+7) * np.exp(x**2 + 7*x - 30)

Nmax = 100
tol = 1e-13
a = 2
b = 4.5
# test f1 '''
x0 = 0.01
[xstar, ier, count] = hybrid_method(f, fprime, fdoubleprime,a,b,tol)
print('the approximate fixed point for x0 = 0.01 is:',xstar)
print('f1(xstar):',f(xstar))
print('Error message reads: ',ier)
print('Number of iterations: ',count)
