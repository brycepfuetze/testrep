import numpy as np
import numpy.linalg as la
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():


    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    N = np.array([5,10,15,20])
    ''' interval'''

    a = -5
    b = 5
    slope_start = fp(a)
    slope_end = fp(b)

    Neval = 1000
    yevalL = np.zeros([Neval+1,4])
    yevalH = np.zeros([Neval+1,4])
    yevalCubic = np.zeros([Neval+1,4])
    yevalCubicClamped = np.zeros([Neval+1,4])
    fex = np.zeros([Neval+1])
    xeval = np.linspace(a,b,Neval+1)
    

    ''' create vector with exact values'''
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])



    for i in range(4):
        print(i)

        ''' create equispaced interpolation nodes'''
        # xint = np.linspace(a,b,N[i]+1)
        # change to Chebychev! 2nd kind!!!
        xint = []
        for j in range(N[i] + 1):
            xint.append(-5 * np.cos(j * np.pi / N[i]))
        xint = np.array(xint)
        
        ''' create interpolation data'''
        yint = np.zeros(N[i]+1)
        ypint = np.zeros(N[i]+1)
        for jj in range(N[i]+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        for kk in range(Neval+1):
            yevalL[kk,i] = eval_lagrange(xeval[kk],xint,yint,N[i])
            yevalH[kk,i] = eval_hermite(xeval[kk],xint,yint,ypint,N[i])


        (M,C,D) = create_natural_spline(yint,xint,N[i])
        (Mc,Cc,Dc) = create_clamped_spline(yint, xint, N[i], slope_start, slope_end)
        
        print('M natural =', M)
        print('M clamped =', Mc)
    #    print('C =', C)
    #    print('D=', D)
        
        yevalCubic[:,i] = eval_cubic_spline(xeval,Neval,xint,N[i],M,C,D)
        yevalCubicClamped[:,i] = eval_cubic_spline(xeval,Neval,xint,N[i],Mc,Cc,Dc)

    trace0 = go.Scatter(x=xint , y=yint, name='Data Points',mode='markers', marker=dict(
        symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yevalL[:,0], name='Interpolated, N=5', yaxis='y1')
    trace2 = go.Scatter(x= xeval, y=yevalL[:,1], name='Interpolated, N=10', yaxis='y1')
    trace3 = go.Scatter(x= xeval, y=yevalL[:,2], name='Interpolated, N=15', yaxis='y1')
    trace4 = go.Scatter(x= xeval, y=yevalL[:,3], name='Interpolated, N=20', yaxis='y1')
    trace5 = go.Scatter(x=xeval, y=fex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function: Lagrange with Chebychev Nodes',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    trace0 = go.Scatter(x=xint , y=yint, name='Data Points',mode='markers', marker=dict(
        symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yevalH[:,0], name='Interpolated, N=5', yaxis='y1')
    trace2 = go.Scatter(x= xeval, y=yevalH[:,1], name='Interpolated, N=10', yaxis='y1')
    trace3 = go.Scatter(x= xeval, y=yevalH[:,2], name='Interpolated, N=15', yaxis='y1')
    trace4 = go.Scatter(x= xeval, y=yevalH[:,3], name='Interpolated, N=20', yaxis='y1')
    trace5 = go.Scatter(x=xeval, y=fex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function: Hermite with Chebychev Nodes',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
    fig = go.Figure(data=data, layout=layout)
    fig.show()


    trace0 = go.Scatter(x=xint , y=yint, name='Data Points',mode='markers', marker=dict(
            symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yevalCubic[:,0], name='Interpolated, N=5', yaxis='y1')
    trace2 = go.Scatter(x= xeval, y=yevalCubic[:,1], name='Interpolated, N=10', yaxis='y1')
    trace3 = go.Scatter(x= xeval, y=yevalCubic[:,2], name='Interpolated, N=15', yaxis='y1')
    trace4 = go.Scatter(x= xeval, y=yevalCubic[:,3], name='Interpolated, N=20', yaxis='y1')
    trace5 = go.Scatter(x=xeval, y=fex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function: Natural Cubic with Chebychev Nodes',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    trace0 = go.Scatter(x=xint , y=yint, name='Data Points',mode='markers', marker=dict(
            symbol='circle-open',size=15),yaxis='y1')
    trace1 = go.Scatter(x= xeval, y=yevalCubicClamped[:,0], name='Interpolated, N=5', yaxis='y1')
    trace2 = go.Scatter(x= xeval, y=yevalCubicClamped[:,1], name='Interpolated, N=10', yaxis='y1')
    trace3 = go.Scatter(x= xeval, y=yevalCubicClamped[:,2], name='Interpolated, N=15', yaxis='y1')
    trace4 = go.Scatter(x= xeval, y=yevalCubicClamped[:,3], name='Interpolated, N=20', yaxis='y1')
    trace5 = go.Scatter(x=xeval, y=fex, name='Exact', yaxis='y1')

    data = [trace0, trace1, trace2, trace3, trace4, trace5]
    layout = go.Layout(title='Comparison of Interpolated and Exact Function: Clamped Cubic with Chebychev Nodes',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    
    


    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)

def create_clamped_spline(yint, xint, N, slope_start, slope_end):
    # Create the right-hand side for the linear system
    b = np.zeros(N + 1)
    h = np.zeros(N + 1)
    
    # Calculate h and b values for internal points
    for i in range(1, N):
        hi = xint[i] - xint[i - 1]
        hip = xint[i + 1] - xint[i]
        b[i] = (yint[i + 1] - yint[i]) / hip - (yint[i] - yint[i - 1]) / hi
        h[i - 1] = hi
        h[i] = hip

    # Adjust b vector for clamped boundary conditions
    h[0] = xint[1] - xint[0]
    h[N] = xint[N] - xint[N - 1]
    b[0] = 6 * ((yint[1] - yint[0]) / h[0] - slope_start) / h[0]
    b[N] = 6 * (slope_end - (yint[N] - yint[N - 1]) / h[N]) / h[N]

    # Create matrix A to solve for the M values
    A = np.zeros((N + 1, N + 1))
    A[0][0] = 2
    A[N][N] = 2

    for j in range(1, N):
        A[j][j - 1] = h[j - 1] / 6
        A[j][j] = (h[j] + h[j - 1]) / 3
        A[j][j + 1] = h[j] / 6

    # Solve for M values
    M = inv(A).dot(b)

    # Create the linear coefficients C and D
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
        C[j] = yint[j] / h[j] - h[j] * M[j] / 6
        D[j] = yint[j + 1] / h[j] - h[j] * M[j + 1] / 6

    return M, C, D
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)            


def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       


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
  

       

driver()        
