import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(true_labels, predict_labels):
    return -np.sum(true_labels * np.log(predict_labels + 1e-8) + (1 - true_labels) * np.log(1 - predict_labels + 1e-8))

train_data = pd.read_csv("./logistic_regression_train_data.csv")
train_features = train_data[['Feature1', 'Feature2']].values
train_features = np.hstack((train_features, np.ones((train_features.shape[0], 1))))
train_labels = train_data['Label'].values

test_data = pd.read_csv("./logistic_regression_test_data.csv")
test_features = test_data[['Feature1', 'Feature2']].values
test_features = np.hstack((test_features, np.ones((test_features.shape[0], 1))))
test_labels = test_data['Label'].values

w = np.random.rand(3)
learning_rate = 0.005
iter_times = 10000

for i in range(1,iter_times+1):
    predict_labels = sigmoid(np.dot(train_features, w))
    loss = cross_entropy_loss(train_labels, predict_labels)
    gradient = np.dot(train_features.T, (predict_labels - train_labels)) / train_features.shape[0]
    w -= learning_rate * gradient
    if i % 100 == 0:
        print("Iteration %d, loss: %.8f" % (i, loss))

test_pred = sigmoid(np.dot(test_features, w))
test_pred_labels = (test_pred >= 0.5).astype(int)
accuracy = np.mean(test_pred_labels == test_labels)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(10, 6))

plt.scatter(train_features[train_labels == 0][:, 0], train_features[train_labels == 0][:, 1], color='blue', label='Class 0')
plt.scatter(train_features[train_labels == 1][:, 0], train_features[train_labels == 1][:, 1], color='red', label='Class 1')

x_values = np.linspace(train_features[:, 0].min(), train_features[:, 0].max(), 100)
y_values = -(w[0] * x_values + w[2]) / w[1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.ylim(-5, 5)
plt.xlim(-3, 3)

plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()