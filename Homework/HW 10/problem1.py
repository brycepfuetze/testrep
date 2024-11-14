import numpy as np
import numpy.linalg as la
import plotly.express as px
import plotly.graph_objects as go
from numpy.linalg import inv 
from numpy.linalg import norm

# I want some Jelly Beans...

x = np.linspace(0,5,10000)
y = np.sin(x)

# Part a: Numerator - cubic, Denominator - cubic
# Part b: Numerator - quadratic, Denominator - quartic
# Part c: Numerator - quartic, Denominator - quadratic

ra = (x - 7/60 * x**3) / (1+ 1/20 * x**2)
rb = (x + 5/9 * x**2) / (1 + 5/9 * x + 1/6 * x**2 + 1/36 * x**3 + 7/360 * x**4)
rc = (x - 7/60 * x**3) / (1+ 1/20 * x**2)
T6 = x - 1/6 * x**3 + 1/120 * x**5

err_a = abs(T6-ra)
err_b = abs(T6-rb)
err_c = abs(T6-rc)

trace0 = go.Scatter(x= x, y=ra, name='(a) Numerator - cubic, Denominator - cubic', yaxis='y1')
trace1 = go.Scatter(x= x, y=rb, name='(b) Numerator - quadratic, Denominator - quartic', yaxis='y1')
trace2 = go.Scatter(x= x, y=rc, name='(c) Numerator - quartic, Denominator - quadratic', yaxis='y1')
trace3 = go.Scatter(x= x, y=y, name='Exact', yaxis='y1')
trace4 = go.Scatter(x= x, y=T6, name='6th order Maclaurin', yaxis='y1')

data = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(title='Pade Polynomial Comparison for sin(x)',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='y',
                          anchor='x'),
                  )
fig = go.Figure(data=data, layout=layout)
fig.show()

trace0 = go.Scatter(x= x, y=err_a, name='(a) Numerator - cubic, Denominator - cubic', yaxis='y1')
trace1 = go.Scatter(x= x, y=err_b, name='(b) Numerator - quadratic, Denominator - quartic', yaxis='y1')
trace2 = go.Scatter(x= x, y=err_c, name='(c) Numerator - quartic, Denominator - quadratic', yaxis='y1')

data = [trace0, trace1, trace2]
layout = go.Layout(title='Pade Polynomial Comparison for sin(x)',
                  height=600,
                  xaxis=dict(
                      title='x'),
                  yaxis=dict(
                          title='error',
                          anchor='x',
                          type = 'log'),
                  )
fig = go.Figure(data=data, layout=layout)
fig.show()