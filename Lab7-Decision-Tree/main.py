import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gini_impurity(labels):
    classes, counts = np.unique(labels, return_counts=True)
    prob = counts / len(labels)
    return 1 - np.sum(prob**2)


def find_best_split(features, labels):
    min_gini = float('inf')
    best_split = None

    for feature in range(features.shape[1]):
        feature_min, feature_max = np.min(features[:, feature]), np.max(features[:, feature])
        split_values = np.linspace(feature_min, feature_max, 100)

        for split_value in split_values:
            left_set = labels[features[:, feature] < split_value]
            right_set = labels[features[:, feature] >= split_value]

            gini = len(left_set) * gini_impurity(left_set) + len(right_set) * gini_impurity(right_set)

            if gini < min_gini:
                min_gini = gini
                best_split = (feature, split_value)

    return best_split
    

def build_tree(features, labels, depth=0, max_depth=5):
    if len(np.unique(labels)) == 1:
        return labels[0]
    if depth >= max_depth:
        return np.argmax(np.bincount(labels))
    
    best_split = find_best_split(features, labels)
    if best_split is None:
        return np.argmax(np.bincount(labels))
    
    feature, split_value = best_split
    left_set = features[:, feature] < split_value
    right_set = ~left_set

    left_tree = build_tree(features[left_set], labels[left_set], depth+1, max_depth)
    right_tree = build_tree(features[right_set], labels[right_set], depth+1, max_depth)

    return (feature, split_value, left_tree, right_tree)
    

def predict(tree, feature):
    if isinstance(tree, (np.int64,np.float64)):
        return tree
    
    feature_index, split_value, left_tree, right_tree = tree

    if feature[feature_index] < split_value:
        return predict(left_tree, feature)
    else:
        return predict(right_tree, feature)


train_data = pd.read_csv("./logistic_regression_train_data.csv")
train_features = train_data[['Feature1', 'Feature2']].values
train_labels = train_data['Label'].values

test_data = pd.read_csv("./logistic_regression_test_data.csv")
test_features = test_data[['Feature1', 'Feature2']].values
test_labels = test_data['Label'].values

train_features = np.hstack((train_features, np.ones((train_features.shape[0], 1))))
test_features = np.hstack((test_features, np.ones((test_features.shape[0], 1))))

decision_tree = build_tree(train_features, train_labels)

predictions = np.array([predict(decision_tree, feature) for feature in test_features])
accuracy = np.mean(predictions == test_labels)
print(f"Accuracy: {accuracy:.4f}")

plt.figure(figsize=(10, 6))

plt.scatter(train_features[train_labels == 0][:, 0], train_features[train_labels == 0][:, 1], color='blue', label='Class 0')
plt.scatter(train_features[train_labels == 1][:, 0], train_features[train_labels == 1][:, 1], color='red', label='Class 1')

x_min, x_max = train_features[:, 0].min(), train_features[:, 0].max()
y_min, y_max = train_features[:, 1].min(), train_features[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]
predictions_grid = np.array([predict(decision_tree, sample) for sample in grid])
predictions_grid = predictions_grid.reshape(xx.shape)

plt.contourf(xx, yy, predictions_grid, alpha=0.3, cmap='coolwarm')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.title('Decision Tree Classifier')
plt.show()