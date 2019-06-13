import numpy as np


# Make test data for linear regression
def linear(n: int, dim: int,
           rand_bound: float = 10., noisy: bool = False) -> tuple:
    assert 0 < dim < n and rand_bound > 0
    x = np.random.uniform(-rand_bound, rand_bound, size=(n, dim))
    w = np.random.uniform(-rand_bound, rand_bound, size=(dim, 1))
    b = np.random.uniform(-rand_bound, rand_bound)
    y = (np.matmul(x, w) + b).reshape(-1)
    # Add gaussian noise to y
    if noisy:
        mu, sigma = 0, 0.1
        y += np.random.normal(mu, sigma, y.shape)
    return x, w, b, y


# Make test data for logistic regression
def logistic(n: int, dim: int,
             rand_bound: float = 10., noisy: bool = False) -> tuple:
    def sigmoid(data: np.ndarray):
        res = 1 / (1 + np.exp(-data))
        return res

    x, w, b, y = linear(n, dim, rand_bound, noisy)
    label = sigmoid(y)
    label[label < 0.5] = 0
    label[label >= 0.5] = 1
    return x, w, b, label


# Make test data for perceptron
def perceptron(n: int, dim: int,
               rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = linear(n, dim, rand_bound, noisy)
    label = np.sign(y)
    # Additional: 0 -> 1 as we only have labels in (-1, 1)
    label[label == 0] = 1
    return x, w, b, label


# Make test data for SVM: same as perceptron
svm = perceptron
