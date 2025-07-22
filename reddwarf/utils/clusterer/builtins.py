from typing import TYPE_CHECKING, cast, Optional
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score

from reddwarf.exceptions import try_import
from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.sklearn.model_selection import GridSearchNonCV
from .registry import register_clusterer

if TYPE_CHECKING:
    import hdbscan as hdbscan_module


def _to_range(r) -> range:
    """
    Creates an inclusive range from a list, tuple, or int.

    Examples:
        _to_range(2) # [2, 3]
        _to_range([2, 5]) # [2, 3, 4, 5]
        _to_range((2, 5)) # [2, 3, 4, 5]
    """
    if isinstance(r, int):
        start = end = r
    elif isinstance(r, (tuple, list)) and len(r) == 2:
        start, end = r
    else:
        raise ValueError("Expected int or a 2-element tuple/list")

    return range(start, end + 1)  # inclusive


class PolisBestKMeans:
    """
    A clusterer that automatically finds the best k-means clustering using silhouette scores.

    This class provides a scikit-learn-like interface while handling the k-selection
    internally using grid search and silhouette scoring.
    """

    def __init__(self, k_bounds: Optional[list] = None, init: str = "polis",
                 init_centers: Optional[list] = None, random_state: Optional[int] = None):
        self.k_bounds = k_bounds or [2, 5]
        self.init = init
        self.init_centers = init_centers
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_k_ = None
        self.best_score_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X: NDArray):
        """Fit the clusterer and find the optimal number of clusters using silhouette scores."""
        param_grid = {
            "n_clusters": _to_range(self.k_bounds),
        }

        def scoring_function(estimator, X_data):
            labels = estimator.fit_predict(X_data)
            return silhouette_score(X_data, labels)

        search = GridSearchNonCV(
            param_grid=param_grid,
            scoring=scoring_function,
            estimator=PolisKMeans(
                init=self.init,
                init_centers=self.init_centers,
                random_state=self.random_state,
            ),
        )

        search.fit(X)

        self.best_k_ = search.best_params_['n_clusters']
        self.best_score_ = search.best_score_
        self.best_estimator_ = search.best_estimator_

        if self.best_estimator_:
            self.labels_ = self.best_estimator_.labels_
            # Expose cluster_centers_ for compatibility with existing code
            if hasattr(self.best_estimator_, 'cluster_centers_'):
                self.cluster_centers_ = self.best_estimator_.cluster_centers_

        return self

    def fit_predict(self, X: NDArray):
        """Fit the clusterer and return cluster labels."""
        self.fit(X)
        return self.labels_


@register_clusterer("kmeans")
def make_kmeans(**kwargs):
    """
    Create a PolisBestKMeans clusterer that automatically finds optimal k.

    This is the default k-means implementation that uses silhouette scores
    to determine the optimal number of clusters.
    """
    defaults: dict = dict(
        k_bounds=[2, 5],
        init="polis",
        init_centers=None,
        random_state=None,
    )
    defaults.update(kwargs)
    return PolisBestKMeans(**defaults)


@register_clusterer("hdbscan")
def make_hdbscan(**kwargs) -> "hdbscan_module.HDBSCAN":
    """
    Create an HDBSCAN clusterer with default parameters.
    """
    hdbscan = try_import("hdbscan", extra="alt-algos")
    if TYPE_CHECKING:
        hdbscan = cast("hdbscan_module", hdbscan)

    defaults: dict = dict(
        min_cluster_size=5,
    )
    defaults.update(kwargs)
    return hdbscan.HDBSCAN(**defaults)
