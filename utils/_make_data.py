import numpy as np

"""Make data for test purpose.
For linear models:
    Shape of x: (n, dim)
    Shape of w: (n,)
    Shape of y: (n,)
"""


# Linear Models
# Make test data for linear regression
def linear(n: int, dim: int,
           rand_bound: float = 10., noisy: bool = False) -> tuple:
    assert 0 < dim < n and rand_bound > 0
    x = np.random.uniform(-rand_bound, rand_bound, size=(n, dim))
    w = np.random.uniform(-rand_bound, rand_bound, size=dim)
    b = np.random.uniform(-rand_bound, rand_bound)
    y = (np.matmul(x, w) + b).reshape(-1)
    # Add gaussian noise to y
    if noisy:
        mean, var = 0, 0.1
        y += np.random.normal(mean, var, y.shape)
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


# Make test data for SVM, which is same as perceptron
svm = perceptron


# Non-linear models
# Make test data for bayes
def bayes(n: int, dim: int,
          rand_bound: float = 10.,
          class_count: int = 2,
          linear_: bool = False,
          noisy: bool = False) -> tuple:
    def __get_random_cov(d: int) -> np.ndarray:
        _cov = np.random.uniform(size=(d, d))
        # Make matrix positive semi-definite
        _cov = np.matmul(_cov, _cov.T)
        return _cov

    # Generate class labels
    assert 0 < dim < n and rand_bound > 0
    assert 1 < class_count < n
    # Generate mean, co-variance matrices
    mean_dict = {}
    cov_dict = {}
    # Generate labels first
    label = np.random.choice(class_count, size=n)
    # Use same covariance matrix if linear seperable
    cov_all = __get_random_cov(dim)
    for c in range(class_count):
        mean = np.random.uniform(-rand_bound, rand_bound, size=dim)
        mean_dict[c] = mean
        if linear_:
            cov = cov_all
        else:
            cov = __get_random_cov(dim)
        cov_dict[c] = cov
    # Generate x
    x = [np.random.multivariate_normal(mean_dict[c], cov_dict[c])
         for c in label]
    x = np.array(x)
    # Add some noise to x
    if noisy:
        mean, var = 0, 0.1
        x += np.random.uniform(mean, var, size=x.shape)
    return x, mean_dict, cov_dict, label


# Make test data for k-nearest-neighbor, same as bayes
k_nearest = bayes
