import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import scipy.linalg
import time
import plotly as plt
import plotly.graph_objects as go

def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
  
     x = scila.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)

     N = [100,500,1000,2000,4000,5000]
     scilatime = np.zeros(6)
     LUfactime = np.zeros(6)
     LUsoltime = np.zeros(6)

     for i, n in enumerate(N):
          b = np.random.rand(n,1)
          A = np.random.rand(n,n)

          start_time = time.perf_counter()
          x = scila.solve(A,b)
          end_time = time.perf_counter()
          scilatime[i] = end_time - start_time
          print(f'Sci LA for N = {n} is ', end_time - start_time)

          start_time = time.perf_counter()
          lu, piv = scipy.linalg.lu_factor(A)
          end_time = time.perf_counter()
          LUfactime[i] = end_time - start_time
          print(f'LU factor for N = {n} is ', end_time - start_time)

          start_time = time.perf_counter()
          x = scipy.linalg.lu_solve((lu, piv), b)
          end_time = time.perf_counter()
          LUsoltime[i] = end_time - start_time
          print(f'LU solve for N = {n} is ', end_time - start_time)

     
     print(r)

     trace0 = go.Scatter(x= N, y=scilatime, name='SciLA Time', yaxis='y1')
     trace1 = go.Scatter(x= N, y=LUfactime, name='LU Factorization Time',line=dict(dash='dash'), yaxis='y1')
     trace2 = go.Scatter(x= N, y=LUsoltime, name='LU Solve Time',line=dict(dash='dash'), yaxis='y1')
     trace3 = go.Scatter(x= N, y=LUfactime+LUsoltime, name='Factorization + Solve', yaxis='y1')

     data = [trace0, trace1, trace2, trace3]
     layout = go.Layout(title='Time for System Solves Size N',
                  height=600,
                  xaxis=dict(
                      title='Size N'),
                  yaxis=dict(
                          title='Time [s]',
                          anchor='x'),
                  )
     fig = go.Figure(data=data, layout=layout)
     fig.show()


     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     
          
  
# if __name__ == '__main__':
#       # run the drivers only if this is called from the command line
#       driver()       
driver()