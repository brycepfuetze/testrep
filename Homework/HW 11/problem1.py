import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import scipy


def driver():
    
    f = lambda s: 1/(1+s**2)
    a = -5
    b =  5
    
    # exact integral
    I_ex = 2.74680153389003
    [I_ex6, abserr6, infodict6] = scipy.integrate.quad(f, a, b, full_output=1, epsabs=10e-6)
    [I_ex4, abserr4, infodict4] = scipy.integrate.quad(f, a, b, full_output=1, epsabs=10e-4)
    print('Scipy QUAD evaluation for tol=10e-6 ', I_ex6)
    print('Absolute Error =  ', abserr6)
    print('n = ', next(iter(infodict6.values())))
    print('Scipy QUAD evaluation for tol=10e-4 ', I_ex4)
    print('Absolute Error =  ', abserr4)
    print('n = ', next(iter(infodict4.values())))    
    
#    N =100
#    ntest = np.arrange(0,N,step=2)
    
#    errorT = np.zeros(len(ntest))
#    errorS = np.zeros(len(ntest))
    
#    for j in range(0,len(ntest)):
#        n = ntest[j]

# for simpson's n must be even.        
# n+1 = number of pts.
    n = 510
    I_trap = CompTrap(a,b,1291,f)
    print('I_trap= ', I_trap)
    
    err = abs(I_ex-I_trap)   
    
    print('absolute error = ', err)
    print('n = ', 1291)      
    
    I_simp = CompSimp(a,b,108,f)

    print('I_simp= ', I_simp)
    
    err = abs(I_ex-I_simp)   
    
    print('absolute error = ', err)   
    print('n = ', 108)    

        
def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    
    return I_trap     

def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # even part 
         I_simp = I_simp+2*f(xnode[2*j])
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp
    
    return I_simp     


    
    
driver()    
