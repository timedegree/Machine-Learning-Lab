import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train_data = pd.read_csv("./logistic_regression_train_data.csv")
train_features = train_data[['Feature1', 'Feature2']].values
train_labels = train_data['Label'].values

test_data = pd.read_csv("./logistic_regression_test_data.csv")
test_features = test_data[['Feature1', 'Feature2']].values
test_labels = test_data['Label'].values

mean_0 = np.mean(train_features[train_labels == 0], axis=0)
mean_1 = np.mean(train_features[train_labels == 1], axis=0)

S_1 = np.dot((train_features[train_labels == 0] - mean_0).T, train_features[train_labels == 0] - mean_0) 
S_2 = np.dot((train_features[train_labels == 1] - mean_1).T, train_features[train_labels == 1] - mean_1)
S_w = S_1 + S_2

w = np.dot(np.linalg.inv(S_w), mean_0 - mean_1)
b = - (np.dot(mean_0.T, np.dot(np.linalg.inv(S_w),mean_0)) - np.dot(mean_1.T, np.dot(np.linalg.inv(S_w),mean_1)))/2

test_pred = sigmoid(np.dot(test_features, w) + b)
test_pred_labels = (test_pred < 0.5).astype(int)
accuracy = np.mean(test_pred_labels == test_labels)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(10, 6))

plt.scatter(train_features[train_labels == 0][:, 0], train_features[train_labels == 0][:, 1], color='blue', label='Class 0')
plt.scatter(train_features[train_labels == 1][:, 0], train_features[train_labels == 1][:, 1], color='red', label='Class 1')

x_values = np.linspace(train_features[:, 0].min(), train_features[:, 0].max(), 100)
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.ylim(-5, 5)
plt.xlim(-3, 3)

plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title('Liner Discriminant Analysis Decision Boundary')
plt.show()