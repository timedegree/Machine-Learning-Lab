import numpy as np
import matplotlib.pyplot as plt

with open("x-y.txt") as file:
    x = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])
    y = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])

A = np.vstack([x, np.ones(len(x))]).T
w = np.random.randn(2)
learning_rate = 0.0011
iter_times = 10000

def loss(A,w,y):
    return np.mean((A.dot(w) - y)**2)

trajectory = []

print(w)
for i in range(1,iter_times+1):
    grad = 2 * A.T.dot(A.dot(w) - y) / len(y)
    w -= learning_rate * grad
    if i%200 == 0:
        trajectory.append(w.copy())
    print(w)

trajectory = np.array(trajectory)

print(w)

w0 = np.linspace(10,18,100)
w1 = np.linspace(0,220,100)
W0, W1 = np.meshgrid(w0, w1)
Z = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        Z[i,j] = loss(A,[W0[i,j],W1[i,j]],y)

plt.contour(W0, W1, Z, 100)
plt.plot(trajectory[:,0], trajectory[:,1], 'ro-', markersize=3, linewidth=1, color='orange')
plt.xlabel('w0')
plt.ylabel('w1')
plt.colorbar()
plt.show()