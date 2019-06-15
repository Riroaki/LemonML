import numpy as np
from supervised._basics import Model


class Bayes(Model):
    """Bayes model, multi-class (or binary) classifier.
    Bayes models include Gaussian, Multinomial, Bernoulli,
    however here I only implemented Gaussian.
    """

    def __init__(self):
        self._prior_dict = None
        self._mean_dict = None
        self._cov_dict = None
        self._cov_all = None
        self._p = None

    def fit(self, x: np.ndarray, label: np.ndarray, **kwargs) -> np.float:
        assert x.shape[0] == label.shape[0]
        n, p = x.shape
        if self._mean_dict is None or self._cov_dict is None \
                or self._prior_dict is None or self._p != p:
            self._prior_dict = {}
            self._mean_dict = {}
            self._cov_dict = {}
            self._p = p

        # Calculate mean and co-variance matrix for each class
        all_class = np.unique(label)
        for c in all_class:
            group = x[label == c]
            mean, cov = self.__param_gaussian(group)
            self._prior_dict[c] = group.shape[0] / n
            self._mean_dict[c] = mean
            self._cov_dict[c] = cov

        # Calculate the whole co-variance matrix
        _, cov = self.__param_gaussian(x)
        self._cov_all = cov

        # Calculate loss on x
        _, loss = self.evaluate(x, label)
        return loss

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        assert self._cov_dict is not None and self._mean_dict is not None
        assert self._cov_all is not None
        assert self._p == x.shape[1]
        # Default: non-linear classifier
        linear = False
        if 'linear' in kwargs:
            assert isinstance(kwargs['linear'], bool)
            linear = kwargs['linear']

        # Calculate posterior propability for each class
        # All class share a same co-variance matrix if linear == True
        prob, label_list = [], []
        for c, mean in self._mean_dict.items():
            if linear:
                cov = self._cov_all
            else:
                cov = self._cov_dict[c]
            prior = self._prior_dict[c]
            current_prob = self.__posterior_gaussian(x, prior, mean, cov)
            prob.append(current_prob)
            label_list.append(c)
        # Get index of class having maximum probability for each x
        pred_val = np.argmax(prob, axis=0)
        label_list = np.array(label_list)
        pred_label = label_list[pred_val]
        return pred_label

    def evaluate(self, x: np.ndarray, label: np.ndarray, **kwargs) -> tuple:
        pred_label = self.predict(x, **kwargs)
        # Calculate 0-1 loss
        loss = np.count_nonzero(pred_label - label)
        # Use loss to calculate precision
        precision = 1 - loss / x.shape[0]
        return precision, loss

    @staticmethod
    def __param_gaussian(x: np.ndarray) -> tuple:
        """Estimate mean and variance."""
        mean = x.mean(axis=0)
        diff = x - mean
        cov = np.matmul(diff.T, diff) / x.shape[0]
        return mean, cov

    @staticmethod
    def __posterior_gaussian(x: np.ndarray, prior: np.float,
                             mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Calculate posterior probability P(wi | x)."""

        # Calculate likelihood probability:
        # P(xj | wi) ~ 1 / sqrt(det(cov))
        # * exp(-0.5 * (xj - mean)^T * cov^(-1) * (xi - mean))
        diff = x - mean
        coef = np.power(np.linalg.det(cov), -0.5)
        inv = np.linalg.pinv(cov)
        # Get exponent for xj (0 < j < n)
        exponents = np.apply_along_axis(
            lambda row: np.float(np.matmul(row, inv).dot(row)), 1, diff)
        likelihood = coef * exponents
        # Posterior = prior * likelihood / evidence (omitted)
        posterior = prior * likelihood
        return posterior
