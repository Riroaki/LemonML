import numpy as np
from ._base import ClusteringModel


class KMeans(ClusteringModel):
    """K Means clustering model."""

    @staticmethod
    def clustering(x: np.ndarray, k: int, **kwargs) -> tuple:
        """A fast implementation of k means clustering.
        Credit to Xinlei Chen, Deng Cai
        for more details, please refer to:
        http://www.cad.zju.edu.cn/home/dengcai/Data/Clustering.html
        """
        x = x.astype(float)
        n = x.shape[0]
        ctrs = x[np.random.permutation(x.shape[0])[:k]]
        idx = np.ones(n)
        x_square = np.expand_dims(np.sum(np.multiply(x, x), axis=1), 1)

        while True:
            distance = -2 * np.matmul(x, ctrs.T)
            distance += x_square
            distance += np.expand_dims(np.sum(ctrs * ctrs, axis=1), 0)
            new_idx = distance.argmin(axis=1)
            if np.array_equal(idx, new_idx):
                break
            idx = new_idx
            ctrs = np.zeros(ctrs.shape)
            for i in range(k):
                ctrs[i] = np.average(x[idx == i], axis=0)

        return ctrs, idx
