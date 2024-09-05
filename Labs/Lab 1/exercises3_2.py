import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,9,10)
y = np.arange(10)

# print('x is ', x)
# print('y is ', y)

print('he first three entries of x are', x[0], x[1], x[2])

w = 10**(-np.linspace(1,10,10))
x = np.linspace(1, len(w), len(w))
print(w)
print(x)

plt.semilogy(x, w)
plt.xlabel('x')
plt.ylabel('w')
plt.title('Exercise 3.2.4')
plt.show()

s = 3*w
plt.semilogy(x, w)
plt.semilogy(x, s)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['w','s'])
plt.title('Exercise 3.2.5')
plt.show()