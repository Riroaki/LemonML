import numpy as np
from ._base import ClusteringModel


class KMeans(ClusteringModel):
    """K Means clustering model."""

    def __init__(self):
        self._iter_bound = 1e-6
        self._max_iter = 100

    def clustering(self, x: np.ndarray, k: int, **kwargs) -> tuple:
        """A fast implementation of k means clustering.
        Credit to Xinlei Chen, Deng Cai
        for more details, please refer to:
        http://www.cad.zju.edu.cn/home/dengcai/Data/Clustering.html
        """
        n, p = x.shape
        # Init centeroids: randomly choose k rows of data, shape = (k, p)
        assert 0 < k < n
        centeroids = self.__init_centeroids(x, k)

        # Start iterations
        iters = 0
        assign = np.uint(np.zeros(n))
        # Get square sum of x
        x_square = self.__calc_square_sum(x)

        while True:
            iters += 1
            # Calculate distances between centeroids and each row:
            # (x - c)^2 = x^2 - 2cx + c^2
            # shape of dists = (n, k)
            ctr_square = self.__calc_square_sum(centeroids)
            dists = -2 * np.matmul(x, centeroids.T)
            dists += ctr_square + x_square
            new_assign = dists.argmin(axis=1)
            # Check whether assign has changed
            if np.array_equal(new_assign, assign):
                break
            assign = new_assign
            # Update centeroids of each cluster
            for i in range(k):
                new_centeroid = x[assign == i].mean(axis=0)
                centeroids[i] = new_centeroid
            # Break if iteration times exceeds bound
            if iters >= self._max_iter:
                break

        # Calculate sum of distance
        dist_sum = self.__calc_dist_sum(x, assign, centeroids)
        return assign, centeroids, dist_sum

    @staticmethod
    def __init_centeroids(x: np.ndarray, k: int) -> np.ndarray:
        unique_rows = np.unique(x, axis=0)
        np.random.shuffle(unique_rows)
        assert unique_rows.shape[0] > k
        centeroids = unique_rows[: k]
        return centeroids

    @staticmethod
    def __calc_square_sum(x: np.ndarray) -> np.ndarray:
        product = np.multiply(x, x)
        square = product.sum(axis=1)
        return square

    @staticmethod
    def __calc_dist_sum(x: np.ndarray, assign: np.ndarray,
                        centeroids: np.ndarray) -> float:
        dist_sum = 0
        k = centeroids.shape[0]
        for i in range(k):
            dist = np.power(x[assign == i] - centeroids[i], 2).sum()
            dist_sum += dist
        return dist_sum
