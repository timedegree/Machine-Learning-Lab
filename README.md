# Machine Learning Lab

My Machine Learning course assignments.

## Content

- **Lab 1**: Learn basic operations with NumPy.
- **Lab 2**: Learn basic operations with Matplotlib.
- **Lab 3**: Perform linear regression to fit a curve for given `x` and `y` scatter points.
- **Lab 4**: Use the data from Lab 3 to fit the curve using the gradient descent method.
- **Lab 4-Extra**: Plot the loss trajectory for the gradient descent method in Lab 4.
- **Lab 5**: Implement a logistic regression model using gradient descent on a given dataset and visualize the training samples along with the decision boundary in 2D.
- **Lab 6**: Use Linear Discriminant Analysis (LDA) to classify the dataset from Lab 5, calculate the sigmoid function for estimation, and plot the decision boundary in 2D.
- **Lab 7**: Use Decision Tree to classify the dataset from Lab 5. Instead of a decision boundary, visualize the classification area in 2D.
- **Lab 8**: Train an SVM model using sklearn's SVC with a linear kernel on 15% noisy training dataset from Lab 5, and visualize both the training samples and classification area in 2D.
- **Lab 9**: Transform FashionMNIST images to 784D vectors, apply PCA for dimensionality reduction to $d$, and use K-NN classification on the reduced test data.
- **Final Lab**: Train a Random Forest model with up to 10 binary trees (max depth 5) using only NumPy and pandas for MNIST dataset classification.

## Usage

```bash
pip install -r requirements.txt
cd lab_directory  # Replace 'lab_directory' with the specific lab folder name
python main.py
```
