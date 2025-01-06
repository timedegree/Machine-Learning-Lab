import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("./mnist_train.csv")
train_labels = train_data['label']
train_features = train_data.drop('label', axis=1)

test_data = pd.read_csv("./mnist_test_part.csv")
test_labels = test_data['label']
test_features = test_data.drop('label', axis=1)

