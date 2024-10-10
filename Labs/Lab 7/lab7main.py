import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objects as go

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
    


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval


x = np.linspace(-1, 1, 1000)
N = 2
h = 2/(N)

xeval = []
for j in np.arange(0,N+1):
    xeval = np.append(xeval, -1 + (j)*h)
    print(j)

print(xeval)

y = lambda x: 1/(1 + 10* (x**2))
yeval = y(xeval)
print(yeval)

vandermonde = np.zeros((len(yeval), N+1))

for j in np.arange(0, len(yeval)):
    for i in np.arange(0, N+1):
        vandermonde[j,i] = xeval[j] ** i

print(vandermonde)

a = scipy.linalg.solve(vandermonde, yeval)
print(a)

def polynomial(p):
    return lambda x: sum(a*x**i for i, a in enumerate(p))
poly = polynomial(a)
#yinterpV = poly(x)





    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)




trace0 = go.Scatter(x=x, y=y(x), name='Original', yaxis='y1')
trace1 = go.Scatter(x=x, y=poly(x), name='Vandermonde', yaxis='y1')
# trace2 = go.Scatter(x=np.log(np.arange(0, len(diff1z))), y=np.log(diff1z.flatten()), name='Z', yaxis='y1')

data = [trace0, trace1]
layout = go.Layout(title='Comparison of Interpolation Methods',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
fig = go.Figure(data=data, layout=layout)
fig.show()