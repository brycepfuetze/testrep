import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import plotly.express as px
import plotly.graph_objects as go

def driver(): 

    f = lambda x: 1 / (1+ (10*x)**2)
    
    N = 25
    a = -1
    b = 1
    
    #h = 2/(N-1)
    #xint = np.zeros(N)
    #for i in range(1,N+1):
    #    xint[i-1] = -1 + (i-1)*h
    h = 2/N

    xint = []
    for j in np.arange(0,N+1):
        xint = np.append(xint, -1 + (j)*h)

    ''' Create interpolation nodes'''
#    xint = np.linspace(a,b,N+1)
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
#    print('coef = ', coef)

# No validate the code
    Neval = 1000    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)

    trace0 = go.Scatter(x=xint , y=yint, name='Data Points',mode='markers', marker=dict(
        symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yeval, name='Interpolated', yaxis='y1')
    trace2 = go.Scatter(x=xeval, y=yex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function N=25',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    
    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

driver()    
