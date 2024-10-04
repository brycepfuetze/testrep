import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objects as go

def forwardDiff(f, s, h):
    forwardDiff = (f(s+h) - f(s)) / h
    return forwardDiff

def centeredDiff(f, s, h):
    centeredDiff = (f(s+h) - f(s-h)) / (2*h)
    return centeredDiff

f = lambda x: np.cos(x)
s = np.pi/2
h = 0.01 * 2. ** (-np.arange(0, 10))

forward = np.zeros((0,0))
centered = np.zeros((0,0))

for i in h:
    forward = np.append(forward, forwardDiff(f, s, i))
    centered = np.append(centered, centeredDiff(f, s, i))

print('Forward: ', forward)
print('Centered: ', centered)

trace0 = go.Scatter(x=np.arange(0, len(h)), y=forward, name='Forward', yaxis='y1')
trace1 = go.Scatter(x=np.arange(0, len(h)), y=centered, name='Centered', yaxis='y1')

data = [trace0, trace1]
layout = go.Layout(title='Definition of Derivative Implementation for cos(x) at pi/2',
                  height=600,
                  xaxis=dict(
                      title='0.01 * 2^(-x)'),
                  yaxis=dict(
                          title='Extimated Derivative Value',
                          anchor='x'),
                  )
fig = go.Figure(data=data, layout=layout)
fig.show()
