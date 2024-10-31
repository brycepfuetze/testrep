import numpy as np
from scipy.integrate import quad

def eval_legendre(x, n):
    if n == 1:
        p = [1,x]
    elif n >= 2:
        p = [1, x]
        for i in range(1,n):
            phi_nplusone = 1/(i+1) * ((2*i+1) * x * p[i] - i*p[i-1])
            p.append(phi_nplusone)
    
    return p

# leg = eval_legendre(0,8)
print((eval_legendre(0,8)))
ran = range(1,2)

for i in range (1,2):
 print(i)
