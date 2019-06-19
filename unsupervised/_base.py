import numpy as np
from abc import abstractmethod, ABC


class UnsupervisedModel(ABC):
    """Abstract class for unsupervised models, majorly for clustering.
    Define basic operations of unsupervised models here:
        clustering.
    """

    @abstractmethod
    def clustering(self, x: np.ndarray, k: int,
                   **kwargs) -> np.ndarray or tuple:
        """Clustering data into some groups.

        :param x: input data to group into clusters, shape = (n, p)
        :param k: assumed count of clusters
        :param kwargs: parameters like k for k-means and spectral,
        and training parameters like iter_bound, max_iter, etc.
        :return: assigned index for input data, shape = (n,)
        """


class Decomposition(ABC):
    """Abstract class for decomposition of data.
    Define basic operations of decomposition here:
        reduce.
    """

    @staticmethod
    @abstractmethod
    def extract_feature(x: np.ndarray, k: int = None) -> tuple:
        """Get reduced eigen vectors and eigen values.

        :param x: input data to reduce dimension, shape = (n, p)
        :param k: number of reduced dimensions, default means p
        :return: sorted eigen values, eigen vectors (large to small)
        shape of eigen vectors: (p, k)
        """

    def reduce(self, x: np.ndarray, k: int = None) -> np.ndarray:
        """Project x into lower dimensions using extracted features.

        :param x: input data to reduce dimension, shape = (n, p)
        :param k: number of reduced dimensions, default means p
        :return: projections of x in reduced dimensions
        shape of projection: (n, p)
        """
        _, feature = self.extract_feature(x, k)
        projection = np.matmul(x, feature)
        return projection
