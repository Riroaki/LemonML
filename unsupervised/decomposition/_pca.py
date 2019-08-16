import numpy as np
from unsupervised.decomposition._base import Decomposition


class PCA(Decomposition):
    """Principle component analysis class."""

    @staticmethod
    def extract_feature(x: np.ndarray, k: int = None) -> tuple:
        _, p = x.shape
        assert k is None or k <= p
        # Get covariance matrix of x
        mean_ = x.mean(axis=0)
        diff_ = x - mean_
        cov_ = np.matmul(diff_.T, diff_)
        # Calculate eigen vectors and values
        eig_value, eig_vector = np.linalg.eig(cov_)
        # Sort eigens by eig value from large to small,
        # and extract top k of them
        idx = np.argsort(-eig_value)
        if k is not None:
            idx = idx[: k]
        eig_value = eig_value[idx]
        eig_vector = eig_vector[:, idx]
        return eig_value, eig_vector
