import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def threeDimIter (f,fx, fy, fz, x0, y0, z0,tol,Nmax):
    count = 0
    x = np.array(x0)
    y = np.array(y0)
    z = np.array(z0)

    d = lambda x, y, z: f(x, y, z) / ((fx(x, y, z))** 2 + (fy(x, y, z))** 2 + (fz(x, y, z))** 2)

    while (count <Nmax):
        count = count +1
        x1 = x0 - d(x0, y0, z0) * fx(x0, y0, z0)
        y1 = y0 - d(x0, y0, z0) * fy(x0, y0, z0)
        z1 = z0 - d(x0, y0, z0) * fz(x0, y0, z0)

        x = np.append(x, x1)
        y = np.append(y, y1)
        z = np.append(z, z1)

        if (np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) <tol):
            xstar = x1
            ystar = y1
            zstar = z1
            ier = 0
            return [np.delete(x, -1), np.delete(y, -1), np.delete(z, -1), xstar, ystar, zstar, ier]
        x0 = x1
        y0 = y1
        z0 = z1

    xstar = x1
    ystar = y1
    zstar = z1
    ier = 1

    return [np.delete(x, -1), np.delete(y, -1), np.delete(z, -1), xstar, ystar, zstar, ier]

def compute_order(x, xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    return [fit, diff1, diff2]



f = lambda x, y, z: x**2 + 4* y**2 + 4* x**2 - 16
fx = lambda x, y, z: 2*x
fy = lambda x, y, z: 8*y
fz = lambda x, y, z: 8*z

x0 = 1
y0 = 1
z0 = 1

tol = 1e-10
Nmax = 100

[x, y, z, xstar, ystar, zstar, ier] = threeDimIter (f,fx, fy, fz, x0, y0, z0,tol,Nmax)

print('the approximate location for f(x,y,z) = 0 with x0=y0=z0=1 is: (',xstar, ', ', ystar, ', ', zstar, ')')
print('f(xstar, ystar, zstar) = ',f(xstar, ystar, zstar))
print('Error message reads:',ier)
print('All x values are: ', x)
print('All y values are: ', y)
print('All z values are: ', z)

[fitx, diff1x, diff2x] = compute_order(x, xstar)
[fity, diff1y, diff2y] = compute_order(y, ystar)
[fitz, diff1z, diff2z] = compute_order(z, zstar)

print('Convergence of X')
print('log(|p_{n+1} -p|) = log(lambda) + alpha*log(|p_n-p|) where:')
print('lambda = ' + str(np.exp(fitx[1])))
print('alpha = ' + str(fitx[0]))

print('Convergence of Y')
print('log(|p_{n+1} -p|) = log(lambda) + alpha*log(|p_n-p|) where:')
print('lambda = ' + str(np.exp(fitx[1])))
print('alpha = ' + str(fity[0]))

print('Convergence of Z')
print('log(|p_{n+1} -p|) = log(lambda) + alpha*log(|p_n-p|) where:')
print('lambda = ' + str(np.exp(fitx[1])))
print('alpha = ' + str(fitz[0]))

trace0 = go.Scatter(x=np.log(np.arange(0, len(diff1x))), y=np.log(diff1x.flatten()), name='X', yaxis='y1')
trace1 = go.Scatter(x=np.log(np.arange(0, len(diff1y))), y=np.log(diff1y.flatten()), name='Y', yaxis='y1')
trace2 = go.Scatter(x=np.log(np.arange(0, len(diff1z))), y=np.log(diff1z.flatten()), name='Z', yaxis='y1')

data = [trace0, trace1, trace2]
layout = go.Layout(title='Convergence of f(x,y,z) = 0 Iteration',
                  height=600,
                  xaxis=dict(
                      title='Log|iterations|'),
                  yaxis=dict(
                          title='Log|error|',
                          anchor='x'),
                  )
fig = go.Figure(data=data, layout=layout)
fig.show()
