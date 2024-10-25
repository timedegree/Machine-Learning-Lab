import numpy as np
import matplotlib.pyplot as plt

with open("x-y.txt") as file:
    x = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])
    y = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])

print(x,y,sep='\n')

A = np.vstack([x, np.ones(len(x))]).T
w = np.linalg.inv((A.T.dot(A))).dot(A.T).dot(y)

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, w[0]*x + w[1], color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()