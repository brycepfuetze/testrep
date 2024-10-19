import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import plotly.express as px
import plotly.graph_objects as go

def driver(): 

    f = lambda x: 1 / (1+ (10*x)**2)
    
    N = 100
    a = -1
    b = 1
    h = 2/N

    xdata = []
    for j in np.arange(0,N+1):
        xdata = np.append(xdata, -1 + (j)*h)

    '''Create interpolation data'''
    ydata = f(xdata)

# Now validate the code
    Neval = 1000    
    xeval = np.linspace(a,b,Neval+1)
    yeval = barycentricLagrange(xdata,ydata, xeval)

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)

    trace0 = go.Scatter(x=xdata , y=ydata, name='Data Points',mode='markers', marker=dict(
        symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yeval, name='Interpolated', yaxis='y1')
    trace2 = go.Scatter(x=xeval, y=yex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function N=100',
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

def  barycentricLagrange(xdata,ydata, xeval):

    # calc weights
    N   = len(xdata)
    wj = np.ones(N)

    for i in range(N):
        for j in range(N):
            if i != j:
                wj[j] = wj[j] * (1 / (xdata[j] - xdata[i]))

    yeval = np.zeros(len(xeval))
    for k in range(len(xeval)):
        if xeval[k] in xdata:
                xindex = np.where(xdata == xeval[k])
                yeval[k] = ydata[xindex]
        else:
            phi = 1
            for i in range(N):
                phi = phi * (xeval[k] - xdata[i])

            sum = 0
            for j in range(N):
                sum = sum + (wj[j] / (xeval[k] - xdata[j]) * ydata[j])
            yeval[k] = phi * sum

    return yeval

driver()    
