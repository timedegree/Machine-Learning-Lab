import numpy as np
import matplotlib.pyplot as plt

with open("x-y.txt") as file:
    x = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])
    y = np.array([float(i) for i in file.readline().rstrip("\n").split(" ")])

print(x,y,sep="\n")

# Formula from Lab3
A_1 = np.vstack([x, np.ones(len(x))]).T
w_1 = np.linalg.inv((A_1.T.dot(A_1))).dot(A_1.T).dot(y)

# Gradient Descent
A_2 = np.vstack([x, np.ones(len(x))]).T
w_2 = np.random.randn(2)
learning_rate = 0.0011
iter_times = 10000

def loss(A,w,y):
    return np.mean((A.dot(w) - y)**2)

for i in range(1,iter_times+1):
    grad = 2 * A_2.T.dot(A_2.dot(w_2) - y) / len(y)
    w_2 -= learning_rate * grad
    if i%100 == 0:
        curr_loss = loss(A_2,w_2,y)
        print(f"epoch {i}: loss = {curr_loss}")
    if i%1000 == 0:
        color = plt.cm.viridis(i / iter_times)
        plt.plot(x, w_2[0] * x + w_2[1], color=color, label=f'Epoch {i} fitted line 2')
    

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, w_1[0]*x + w_1[1], color='red', label='Fitted line 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()