import numpy as np
A = np.array([[12, 10, 4],
              [10, 8, -5],
              [4, -5, 3]], dtype=float)

x = np.array([[12],
              [10],
              [4]], dtype=float)

y = np.array([[12],
              [np.sqrt(116)],
              [0]], dtype=float)

w = (x-y) / np.linalg.norm(x - y)

print(w)

P = np.eye(3,3) - 2 * w * np.transpose(w)
tridiag = np.matmul(np.matmul(P,A),P)

print(P)
print(tridiag)

