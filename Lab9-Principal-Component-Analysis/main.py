import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm

def hymmnos_knn(train_images, train_labels, test_images, k=5, distance_file="distances.npy"):
    if os.path.exists(distance_file):
        print(f"Loading precomputed distances from {distance_file}...")
        distances = np.load(distance_file)
    else:
        print("Computing distances...")
        distances = np.zeros((test_images.shape[0], train_images.shape[0]))
        for i, test_point in enumerate(tqdm(test_images, desc="Calculating distances")):
            distances[i] = np.linalg.norm(train_images - test_point, axis=1)
        np.save(distance_file, distances)
        print(f"Distances saved to {distance_file}.")

    predictions = []
    print("Processing K-NN predictions...")
    for i in tqdm(range(test_images.shape[0]), desc="Predicting"):
        nearest_neighbors = train_labels[np.argsort(distances[i])[:k]]
        prediction = np.bincount(nearest_neighbors).argmax()
        predictions.append(prediction)

    return np.array(predictions)

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_images = train_dataset.data.numpy().reshape(-1, 28*28)
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy().reshape(-1, 28*28)
test_labels = test_dataset.targets.numpy() 

train_images_normalized = (train_images - train_images.mean(axis=0)) / train_images.std(axis=0)
cov_matrix = np.cov(train_images_normalized, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_eigenvalues = np.argsort(eigenvalues)[::-1]

d = 200
contribution_rate = np.sum(eigenvalues[sorted_eigenvalues][:d]) / np.sum(eigenvalues)
#print('choice of the eigenvalues:', eigenvalues[sorted_eigenvalues][:0])
print('contribution rate:', contribution_rate)

selected_eigenvectors = eigenvectors[:, sorted_eigenvalues[:d]]
train_images_reduced = np.dot(train_images_normalized, selected_eigenvectors)

test_images_normalized = (test_images - train_images.mean(axis=0)) / train_images.std(axis=0)
test_images_reduced = np.dot(test_images_normalized, selected_eigenvectors)

predictions = hymmnos_knn(train_images_reduced, train_labels, test_images_reduced, k=10)
accuracy = np.mean(predictions == test_labels)
print('Accuracy:', accuracy)

num_samples = 5
random_indices = np.random.choice(train_images.shape[0], num_samples, replace=False)
original_images = train_images[random_indices]
original_labels = train_labels[random_indices]

reconstructed_images = np.dot(train_images_reduced[random_indices], selected_eigenvectors.T) * train_images.std(axis=0) + train_images.mean(axis=0)

plt.figure(figsize=(12, 6))

for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Original (Label: {original_labels[i]})")
    plt.axis('off')
    
    plt.subplot(2, num_samples, num_samples + i + 1)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()