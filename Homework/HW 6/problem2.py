import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy
import math
from numpy.linalg import inv 
from numpy.linalg import norm
import time


def F(x,y,z):
    return np.array([ x + np.cos(x*y*z) - 1,
                    (1-x)**(1/4) + y + 0.05 * z**2 - 0.15*z - 1,
                    -x**2 -0.1*y**2 + 0.01*y + z -1])
 
def J(x,y,z):
    return np.array([[1 + y*z * np.cos(x*y*z),x*z * np.cos(x*y*z), x*y * np.cos(x*y*z)],
             [ 1/4* ((1-x)**(-3/4)), 1, 0.1*z-0.15],
             [ -2*x, -0.2*y+0.01, 1]])

def newton3D(x, y, z, F, J, tol, Nmax):
    count = 0
 
    F0 = F(x,y,z)
    J0 = J(x,y,z)
 
    while np.linalg.norm(F0,2) > tol and count < Nmax:
        
        # solve linear system
        P = scipy.linalg.solve(J0,-1*F0)

        # extract new values
        x = x + P[0]
        y = y + P[1]
        z = z + P[2]
        F1 = F(x,y,z)

        J1 = J(x,y,z)

        J0 = J1
        F0 = F1
        
        count = count+1
 
    return x, y, z, count

def steepestNewtonHybrid1(x,tol,Nmax):
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier, its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<5e-2:
            ier = 0
            return [x,g1,ier, its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier, its]
def steepestNewtonHybrid2(x, y, z, F, J, tol, Nmax, count):
 
    F0 = F(x,y,z)
    J0 = J(x,y,z)
 
    while np.linalg.norm(F0,2) > tol and count < Nmax:
        
        # solve linear system
        P = scipy.linalg.solve(J0,-1*F0)

        # extract new values
        x = x + P[0]
        y = y + P[1]
        z = z + P[2]
        F1 = F(x,y,z)

        J1 = J(x,y,z)

        J0 = J1
        F0 = F1
        
        count = count+1
 
    return x, y, z, count

###########################################################
#functions:
def evalF(x):

    F = np.zeros(3)
    F[0] = x[0] +math.cos(x[0]*x[1]*x[2])-1.
    F[1] = (1.-x[0])**(0.25) + x[1] +0.05*x[2]**2 -0.15*x[2]-1
    F[2] = -x[0]**2-0.1*x[1]**2 +0.01*x[1]+x[2] -1
    return F

def evalJ(x): 

    J =np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
          [-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier, its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,g1,ier, its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier, its]


Nmax = 100
x0= np.array([0,0,1])
tol = 1e-6

print('Newtons: ')
print('x0 = ', x0)
start_time = time.time()
[x, y, z, its] = newton3D(x0[0], x0[1], x0[2], F, J, tol, Nmax)
end_time = time.time()
elapsed_time = end_time - start_time
print("xstar =  ",x , y, z)
#print("the gradient at xstar is ", gval)
#print("ier is ", ier)
print("# of iterations: ", its)
print(f"Elapsed time: {elapsed_time} seconds")


print('Steepest Descent: ')
print('x0 = ', x0)
start_time = time.time()
[xstar,gval,ier, its] = SteepestDescent(x0,tol,Nmax)
end_time = time.time()
elapsed_time = end_time - start_time
print("xstar =  ",xstar)
print("the gradient at xstar is ", gval)
print("# of iterations: ", its+1)
print(f"Elapsed time: {elapsed_time} seconds")


print('Hybrid: ')
print('x0 = ', x0)
start_time = time.time()
[xstar,gval,ier, its] = steepestNewtonHybrid1(x0,tol,Nmax)
[x, y, z, count] = steepestNewtonHybrid2(xstar[0], xstar[1], xstar[2], F, J, tol, Nmax, its)
end_time = time.time()
elapsed_time = end_time - start_time
print("xstar =  ",x , y, z)
print("Total # of iterations: ", count)
print("Steepest # of iterations: ", its+1)
print("Newton # of iterations: ", count - its-1)
print(f"Elapsed time: {elapsed_time} seconds")
