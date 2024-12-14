from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("./logistic_regression_train_data.csv")
train_features = train_data[['Feature1', 'Feature2']].values
#train_labels = train_data['Label'].values

test_data = pd.read_csv("./logistic_regression_test_data.csv")
test_features = test_data[['Feature1', 'Feature2']].values
test_labels = test_data['Label'].values

clf = svm.SVC(kernel='linear')

noisy_rows = int(0.15 * len(train_data))
train_data.loc[:noisy_rows-1, 'Label'] = 1 - train_data.loc[:noisy_rows-1, 'Label']
train_labels = train_data['Label'].values

clf.fit(train_features, train_labels)

accuracy = clf.score(test_features, test_labels)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(train_features[train_labels == 0][:, 0], train_features[train_labels == 0][:, 1], color='blue', label='Class 0')
plt.scatter(train_features[train_labels == 1][:, 0], train_features[train_labels == 1][:, 1], color='red', label='Class 1')

x_min, x_max = train_features[:, 0].min() - 1, train_features[:, 0].max() + 1
y_min, y_max = train_features[:, 1].min() - 1, train_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('SVM Classification with Noisy Train Dataset Labels (15%)')
plt.show()