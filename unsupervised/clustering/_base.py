import numpy as np
from abc import ABC, abstractmethod


class ClusteringModel(ABC):
    """Abstract class for clustering models.
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
