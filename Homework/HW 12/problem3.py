import numpy as np


def makeA(N):
    A = np.zeros((N,N))
    for i in range(1,N+1):
        for j in range(1,N+1):
            A[i-1,j-1] = 1 / (i + j -1)
    return A

A = makeA(6)
print(A)

n = np.linspace(4,20,5).astype(int)
print(n)

maxIter = 1000
tol = 1e-10
eigval = np.zeros(len(n))


for idx, N in enumerate(n):
    A = makeA(N)
    yk = np.random.rand(A.shape[1])
    yk = yk / np.linalg.norm(yk)
    eigenvalue = 0

    for i in range(maxIter):
        yk1 = np.dot(A, yk)  
        yk1_norm = np.linalg.norm(yk1)  
        yk1 = yk1 / yk1_norm  
        if np.linalg.norm(yk1 - yk) < tol: 
            eigenvalue = yk1_norm
            iter = i
            break
        yk = yk1  

    eigval[idx] = eigenvalue 
    print('N = ', N)
    print('Eigenvalue = ',eigenvalue)
    print('Eigenvector = ',yk1)
    print('After ', iter, ' iterations')

print(eigval)

# now invert A for lowest
for idx, N in enumerate(n):
    A = makeA(N)
    Ainv = np.linalg.inv(A)
    yk = np.random.rand(A.shape[1])
    yk = yk / np.linalg.norm(yk)
    eigenvalue = 0

    for i in range(maxIter):
        yk1 = np.dot(Ainv, yk)  
        yk1_norm = np.linalg.norm(yk1)  
        yk1 = yk1 / yk1_norm  
        if np.linalg.norm(yk1 - yk) < tol: 
            eigenvalue = yk1_norm
            iter = i
            break
        yk = yk1  

    eigval[idx] = eigenvalue 
    print('N = ', N)
    print('Eigenvalue = ',eigenvalue)
    print('Eigenvector = ',yk1)
    print('After ', iter, ' iterations')

print(eigval)

A = np.array([[0, -1],
              [1, 0]], dtype=float)
yk = np.random.rand(A.shape[1])
yk = yk / np.linalg.norm(yk)
eigenvalue = 0

for i in range(maxIter):
        yk1 = np.dot(A, yk)  
        yk1_norm = np.linalg.norm(yk1)  
        yk1 = yk1 / yk1_norm  
        if np.linalg.norm(yk1 - yk) < tol: 
            eigenvalue = yk1_norm
            iter = i
            break
        yk = yk1  

eigval[idx] = eigenvalue 
print('Complex mat')
print(A)
print('Eigenvalue = ',eigenvalue)
print('Eigenvector = ',yk1)
print('After ', iter, ' iterations')
