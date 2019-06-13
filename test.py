import supervised
import utils
import numpy as np
import time


# Test supervised models
def test_linear(n: int, dim: int,
                rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = utils.make_data.linear(n, dim, rand_bound, noisy)
    model = supervised.LinearRegression()
    losses, scores = utils.cross_validate.k_fold(x, y, 10,
                                                 model.fit, model.evaluate,
                                                 shuffle=True)
    return losses, scores


def test_linear_norm_eq(n: int, dim: int,
                        rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = utils.make_data.linear(n, dim, rand_bound, noisy)
    model = supervised.LinearRegression()
    losses, scores = utils.cross_validate.k_fold(x, y, 10,
                                                 model.fit_norm_eq,
                                                 model.evaluate,
                                                 shuffle=True)
    return losses, scores


def test_logistic(n: int, dim: int,
                  rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = utils.make_data.logistic(n, dim, rand_bound, noisy)
    model = supervised.LogisticRegression()
    losses, scores = utils.cross_validate.k_fold(x, y, 10,
                                                 model.fit, model.evaluate,
                                                 shuffle=True)
    return losses, scores


def test_perceptron(n: int, dim: int,
                    rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = utils.make_data.perceptron(n, dim, rand_bound, noisy)
    model = supervised.Perceptron()
    losses, scores = utils.cross_validate.k_fold(x, y, 10,
                                                 model.fit, model.evaluate,
                                                 shuffle=True)
    return losses, scores


def test_svm(n: int, dim: int,
             rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, w, b, y = utils.make_data.svm(n, dim, rand_bound, noisy)
    model = supervised.SVM()
    losses, scores = utils.cross_validate.k_fold(x, y, 10,
                                                 model.fit, model.evaluate,
                                                 shuffle=True)
    return losses, scores


def main():
    # I don't know why but, using rand_bound >= 20,
    # the model would fail to converge.

    seed = int(time.time())
    np.random.seed(seed)
    print('\nLinear Regression\n', test_linear(100, 12))
    print('\nLinear Regression: normal equation\n',
          test_linear_norm_eq(100, 12))
    print('\nLogistic Regression\n', test_logistic(2000, 12))
    print('\nPreceptron\n', test_perceptron(1000, 12))
    print('\nSupport Vector Machine\n', test_svm(1000, 12))


if __name__ == '__main__':
    main()
