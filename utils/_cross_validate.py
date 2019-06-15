import numpy as np
from functools import partial

"""Cross validation.
Shape of input data: (n, dim)
Shape of output y: (n,)
"""


def __split(data: np.ndarray, y: np.ndarray, k: int,
            shuffle: bool = True) -> tuple:
    """Split data into k folds."""
    n = data.shape[0]
    assert n >= k
    fold_size = int(np.ceil(n / k))
    indices = None
    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n, fold_size):
        end_idx = start_idx + fold_size
        if shuffle:
            yield (np.concatenate((data[indices[0: start_idx]],
                                   data[indices[end_idx:]])),
                   np.concatenate((y[indices[0: start_idx]],
                                   y[indices[end_idx:]])),
                   data[indices[start_idx: end_idx]],
                   y[indices[start_idx: end_idx]])
        else:
            yield (data[start_idx: end_idx],
                   y[start_idx: end_idx],
                   np.concatenate((data[0: start_idx], data[end_idx:])),
                   np.concatenate((y[0: start_idx], y[end_idx:])))


def k_fold(data: np.ndarray, y: np.ndarray, k: int,
           fit_func: callable, eval_func: callable,
           shuffle: bool = True) -> tuple:
    """K-fold cross validation."""
    train_losses, valid_scores = [], []
    for (train_data, train_y,
         valid_data, valid_y) in __split(data, y, k, shuffle):
        train_losses.append(fit_func(train_data, train_y))
        valid_scores.append(eval_func(valid_data, valid_y))
    return np.array(train_losses), np.array(valid_scores)


def leave_one_out(data: np.ndarray, y: np.ndarray,
                  fit_func: callable, eval_func: callable,
                  shuffle: bool = True) -> tuple:
    """Leave-one-out cross validation: just use k = n."""
    k = data.shape[0]
    return k_fold(data, y, k, fit_func, eval_func, shuffle)
