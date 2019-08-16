from functools import partial
import numpy as np
import time
import supervised
import utils


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


def test_bayes(n: int, dim: int,
               rand_bound: float = 10.,
               linear_: bool = False,
               noisy: bool = False) -> tuple:
    x, mean_dict, cov_dict, label = utils.make_data.bayes(n, dim, rand_bound,
                                                          3, linear_, noisy)
    model = supervised.Bayes()
    losses, scores = utils.cross_validate.k_fold(x, label, 10,
                                                 model.fit,
                                                 model.evaluate,
                                                 shuffle=True)
    return losses, scores


def test_knn(n: int, dim: int, k: int = 5,
             rand_bound: float = 10., noisy: bool = False) -> tuple:
    x, _, _, label = utils.make_data.k_nearest(n, dim, rand_bound, 4,
                                               noisy=noisy)
    model = supervised.KNearest()
    evaluate = partial(model.evaluate, k=k)
    losses, scores = utils.cross_validate.k_fold(x, label, 10,
                                                 model.fit, evaluate,
                                                 shuffle=True)
    return losses, scores


def test():
    # I don't know why but, using rand_bound >= 20,
    # the model would fail to converge.

    seed = int(time.time())
    np.random.seed(seed)
    print('\nLinear Regression:\n', test_linear(100, 12))
    print('\nLinear Regression: normal equation\n',
          test_linear_norm_eq(100, 12))
    print('\nLogistic Regression:\n', test_logistic(2000, 12))
    print('\nPreceptron:\n', test_perceptron(1000, 12))
    print('\nSupport Vector Machine:\n', test_svm(1000, 12))
    print('\nBayes:\n', test_bayes(1000, 12))
    print('\nK-Nearest-Neighbor:\n', test_knn(1000, 12))


if __name__ == '__main__':
    test()
