import numpy as np
import pandas as pd

def timedegree_gini_impurity(labels):
    classes, counts = np.unique(labels, return_counts=True)
    prob = counts / len(labels)
    return 1 - np.sum(prob**2)

def most_frequent_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]

def find_best_split(features, labels):
    best_gini = float('inf')
    best_split = None

    for feature_idx in range(features.shape[1]):
        thresholds = np.unique(features[:, feature_idx])
        for threshold in thresholds:
            left_indices = features[:, feature_idx] <= threshold
            right_indices = ~left_indices

            left_labels = labels[left_indices]
            right_labels = labels[right_indices]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            gini = (len(left_labels) / len(labels)) * timedegree_gini_impurity(left_labels) + \
                   (len(right_labels) / len(labels)) * timedegree_gini_impurity(right_labels)

            if gini < best_gini:
                best_gini = gini
                best_split = {
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'left_indices': left_indices,
                    'right_indices': right_indices
                }

    return best_split

def build_tree(features, labels, depth=0, max_depth=5):
    print(f"Building tree at depth {depth}")

    if depth == max_depth or len(np.unique(labels)) == 1 or len(labels) < 2:
        return most_frequent_label(labels)

    best_split = find_best_split(features, labels)

    if best_split is None:
        return most_frequent_label(labels)

    feature_idx = best_split['feature_idx']
    threshold = best_split['threshold']

    print(f"Split at feature {feature_idx} with threshold {threshold:.4f}")

    left_features = features[best_split['left_indices']]
    left_labels = labels[best_split['left_indices']]
    right_features = features[best_split['right_indices']]
    right_labels = labels[best_split['right_indices']]

    return {
        'feature_idx': feature_idx,
        'threshold': threshold,
        'left': build_tree(left_features, left_labels, depth + 1, max_depth),
        'right': build_tree(right_features, right_labels, depth + 1, max_depth)
    }

def tree_predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature_idx = tree['feature_idx']
    threshold = tree['threshold']

    if sample[feature_idx] <= threshold:
        return tree_predict(tree['left'], sample)
    else:
        return tree_predict(tree['right'], sample)

def build_forest(features, labels, tree_num=10, max_depth=5):
    trees = []
    data_num = len(features)

    split_size = data_num // tree_num
    indices = np.arange(data_num)
    np.random.shuffle(indices)
    splits = [indices[i * split_size:(i + 1) * split_size] for i in range(tree_num)]

    for i, split_indices in enumerate(splits):
        print(f"Building tree {i + 1}/{tree_num}")
        partition_features = features[split_indices]
        partition_labels = labels[split_indices]
        tree = build_tree(partition_features, partition_labels, max_depth=max_depth)
        trees.append(tree)

    return trees

def forest_predict(forest, sample):
    predictions = [tree_predict(tree, sample) for tree in forest]
    return most_frequent_label(predictions)

def evaluate_model(forest, test_features, test_labels):
    predictions = [forest_predict(forest, sample) for sample in test_features]
    accuracy = np.mean(predictions == test_labels)
    return accuracy

train_data = pd.read_csv("./mnist_train.csv")
shuffled_train_data = train_data.sample(frac=1).reset_index(drop=True)
train_labels = shuffled_train_data['label'].values
train_features = shuffled_train_data.drop('label', axis=1).values

forest = build_forest(train_features, train_labels, tree_num=10, max_depth=5)

test_data = pd.read_csv("./mnist_test_part.csv")
test_labels = test_data['label'].values
test_features = test_data.drop('label', axis=1).values

accuracy = evaluate_model(forest, test_features, test_labels)
print(f"Accuracy: {accuracy:.2%}")
