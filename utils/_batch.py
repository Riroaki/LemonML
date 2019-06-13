import numpy as np


def batch(data: np.ndarray, y: np.ndarray, size: int,
          shuffle: bool = False) -> tuple:
    """Make batches of input and output.

    :param data: input data
    :param y: true values
    :param size: number of data entry per batch
    :param shuffle: whether to get shuffled batches
    :return: batched input, batched y
    """
    n = data.shape[0]
    indices = None
    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n, size):
        end_idx = start_idx + size
        if shuffle:
            yield (data[indices[start_idx: end_idx]],
                   y[indices[start_idx: end_idx]])
        else:
            yield data[start_idx: end_idx], y[start_idx: end_idx]
