import numpy as np

def splineEval(x, y, alpha):
    if x[1] < x[0]:
        xone = x[1]
        xzero = x[0]
        x[1] = xzero
        x[0] = xone

        yone = y[1]
        yzero = y[0]
        y[1] = yzero
        y[0] = yone

    h = x[1] - x[0]

    si = lambda xx: 1/h * (y[0] * (x[1] - xx) + y[1] * (xx - x[0]))
    return si(alpha)

# Lets test this thang
x = np.array([0,1])
y = np.array([0,2])
alpha = 0.25

fofalpha = splineEval(x, y, alpha)
print('f of alpha = ', fofalpha)