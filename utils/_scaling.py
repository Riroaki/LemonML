import numpy as np

"""Scaling: normalize features."""


def std(data: np.ndarray):
    """Standardization: zero mean and unit variance."""
    mean_ = data.mean(axis=0)
    std_ = data.std(axis=0)
    data -= mean_
    data /= std_


def minmax(data: np.ndarray):
    """MinMax-scale: scale each dimension to [0, 1] using min-max values."""
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)
    data -= min_
    data /= (max_ - min_)


def mean(data: np.ndarray):
    """Mean-scale: scale each dimension to [-1, 1] using mean and min-max."""
    min_ = data.min(axis=0)
    max_ = data.max(axis=0)
    mean_ = data.mean(axis=0)
    data -= mean_
    data /= (max_ - min_)


def unit(data: np.ndarray):
    """Unit-scale: scale to vectors with unit length."""
    norm_ = np.sqrt(np.power(data, 2).sum(axis=1).reshape((-1, 1)))
    data /= norm_
