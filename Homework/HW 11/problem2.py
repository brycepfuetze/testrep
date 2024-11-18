import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import scipy


def driver():
    
    f = lambda t: np.cos(1/t) * t
    a = 0.0000000000000001
    b = 1
    
    # exact integral
    I_ex = 0.0181176219806056727055402448509141870536626962026415283993320439
    #[I_ex6, abserr6, infodict6] = scipy.integrate.quad(f, a, b, full_output=1, epsabs=10e-6)
    #[I_ex4, abserr4, infodict4] = scipy.integrate.quad(f, a, b, full_output=1, epsabs=10e-4)
    #print('Scipy QUAD evaluation for tol=10e-6 ', I_ex6)
    #print('Absolute Error =  ', abserr6)
    #print('n = ', next(iter(infodict6.values())))
    #print('Scipy QUAD evaluation for tol=10e-4 ', I_ex4)
    #print('Absolute Error =  ', abserr4)
    #print('n = ', next(iter(infodict4.values())))    
    
#    N =100
#    ntest = np.arrange(0,N,step=2)
    
#    errorT = np.zeros(len(ntest))
#    errorS = np.zeros(len(ntest))
    
#    for j in range(0,len(ntest)):
#        n = ntest[j]

# for simpson's n must be even.        
# n+1 = number of pts.
    #n = 4
    #I_trap = CompTrap(a,b,1291,f)
    #print('I_trap= ', I_trap)
    
    #err = abs(I_ex-I_trap)   
    
    #print('absolute error = ', err)
    #print('n = ', 1291)      
    
    I_simp = CompSimp(a,b,2,f)

    print('I_simp= ', I_simp)
    
    err = abs(I_ex-I_simp)   
    
    print('absolute error = ', err)   
    print('N = ', 2)    

        
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
