import numpy as np
from sklearn.base import ClusterMixin
from sklearn.preprocessing import Normalizer


class NormmalizedSKLearnClustering:
    def __init__(
        self,
        tool: ClusterMixin,
        n_clusters: int,
    ):
        self.tool = tool
        self.normalizer = Normalizer()
        self.n_clusters = n_clusters

    def fit(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.tool.fit(self.normalizer.transform(X))

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.tool.predict(self.normalizer.transform(X))
