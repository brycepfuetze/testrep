#import mypkg.my2DPlotB
import plotly.express as px
import numpy as np
import math
from numpy.linalg import inv 


def driver():
    
    f = lambda x: 1 / (1+ (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    plt = px.line(x=xeval,y=fex)
    plt.show()   
     
    plt1 = px.line(x=xeval,y=yeval)
    plt1.show()  
     
    err = abs(yeval-fex)
    plt2 = px.line(x=xeval,y=err)
    plt2.show()            

def splineEval(x, y, alpha):
    h = x[1] - x[0]
    return 1/h * (y[0] * (x[1] - alpha) + y[1] * (alpha - x[0]))

    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
      a1= xint[jint]
      fa1 = f(a1)
      b1 = xint[jint+1]
      fb1 = f(b1)

        
      for kk in range(int(jint * Neval/Nint),int((jint+1) * Neval/Nint)):
        yeval[kk] = splineEval(np.array([a1, b1]), np.array([fa1, fb1]), xeval[kk])
        #use your line evaluator to evaluate the lines at each of the points 
        #in the interval'''
        #yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
        #the points (a1,fa1) and (b1,fb1)'''
    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
