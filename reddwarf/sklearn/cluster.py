from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_random_state, check_array

from reddwarf.sklearn.model_selection import GridSearchNonCV


def _to_range(r) -> range:
    """
    Creates an inclusive range from a list, tuple, or int.

    Examples:
        _to_range(2) # [2]
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


class PolisKMeans(KMeans):
    """
    A modified version of scikit-learn's KMeans that allows partial initialization
    with user-supplied cluster centers and custom fallback strategies.

    This subclass extends `sklearn.cluster.KMeans` with additional features
    around centroid initialization. Outside the behavior documented, it retains
    all other parameters and behavior from the base KMeans implementation.

    Parameters
    ----------

    init : {'k-means++', 'random', 'polis'}, default='k-means++'
        Strategy to initialize any missing cluster centers if `init_centers` is
        not fully specified. The strategies are:

        - 'k-means++': Smart centroid initialization (same as scikit-learn default)
        - 'random': Random selection of initial centers from the data (same as scikit-learn)
        - 'polis': Selects the first unique data points in `X` as initial centers.
            - This strategy is deterministic for any stable set of `X`, while
            determinism in the other strategies depends on `random_state`.

        !!! note
            Unlike `KMeans` parent class, we prevent passing `ndarray` args
            here, and expect `init_centers` to handle that use-case.

    init_centers : ndarray of shape (n_clusters, n_features), optional
        Initial cluster centers to use. May contain fewer (or more) than `n_clusters`:

        - If more, the extras will be trimmed
        - If fewer, the remaining will be filled using the `init` strategy

    Attributes
    ----------

    init_centers_used_ : ndarray of shape (n_clusters, n_features)
        The full array of initial cluster centers actually used to initialize the algorithm,
        including both `init_centers` and any centers generated from the `init` strategy.

    See Also
    --------

    `sklearn.cluster.KMeans` : Original implementation with full parameter list.
    """
    def __init__(
        self,
        n_clusters=8,
        init="k-means++",  # or 'random', 'polis'
        init_centers: Optional[ArrayLike] = None,  # array-like, optional
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,  # will override via set_params, with our center selection logic below
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )
        self._init_strategy = init
        self.init_centers = init_centers
        self.init_centers_used_ = None

    def _generate_centers(self, X, x_squared_norms, n_to_generate, random_state) -> np.ndarray:
        if not isinstance(self._init_strategy, str):
            raise ValueError("Internal error: _strategy must be a string.")

        if self._init_strategy == "k-means++":
            centers, _ = kmeans_plusplus(
                X, n_clusters=n_to_generate,
                random_state=random_state,
                x_squared_norms=x_squared_norms
            )
        elif self._init_strategy == "random":
            indices = random_state.choice(X.shape[0], n_to_generate, replace=False)
            centers = X[indices]
        elif self._init_strategy == "polis":
            unique_X = np.unique(X, axis=0)
            if len(unique_X) < n_to_generate:
                raise ValueError("Not enough unique rows in X for 'polis' strategy.")
            centers = unique_X[:n_to_generate]
        else:
            raise ValueError(f"Unsupported init strategy: {self._init_strategy}")
        return centers

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32]) # type:ignore
        random_state = check_random_state(self.random_state)
        x_squared_norms = np.sum(X ** 2, axis=1)

        # Determine init_centers_used_
        if self.init_centers is not None:
            init_array = np.array(self.init_centers)
            if init_array.ndim != 2 or init_array.shape[1] != X.shape[1]:
                raise ValueError("init_centers must be of shape (n, n_features)")

            n_given = init_array.shape[0]
            if n_given > self.n_clusters:
                init_array = init_array[:self.n_clusters]
            elif n_given < self.n_clusters:
                needed = self.n_clusters - n_given
                extra = self._generate_centers(X, x_squared_norms, needed, random_state)
                init_array = np.vstack([init_array, extra])
            self.init_centers_used_ = init_array.copy()
        else:
            self.init_centers_used_ = self._generate_centers(
                X, x_squared_norms, self.n_clusters, random_state
            )

        # Override the init param passed to sklearn with actual centers.
        # We take control of the initialization strategy (`k-means++`, `random`,
        # `polis`, etc) in our own code.
        super().set_params(init=self.init_centers_used_)
        return super().fit(X, y=y, sample_weight=sample_weight)



class PolisKMeansDownsampler(BaseEstimator, TransformerMixin):
    """
    A transformer that fits PolisKMeans and returns cluster centers as downsampled data.

    This supports mimicking "base clusters" from the Polis platform and enables
    use in sklearn pipelines where intermediate steps implement both fit and transform.

    Parameters
    ----------
    n_clusters : int, default=100
        Number of clusters to form
    random_state : int, RandomState instance or None, default=None
        Random state for reproducible results
    init : {'k-means++', 'random', 'polis'}, default='k-means++'
        Initialization strategy
    init_centers : array-like of shape (n_clusters, n_features), optional
        Initial cluster centers
    """

    def __init__(
        self,
        n_clusters: int = 100,
        random_state: Optional[int] = None,
        init: str = "k-means++",
        init_centers: Optional[ArrayLike] = None,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init
        self.init_centers = init_centers
        self.kmeans_ = None

    def fit(self, X, y=None) -> 'PolisKMeansDownsampler':
        self.kmeans_ = PolisKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            init=self.init,
            init_centers=self.init_centers,
        )
        self.kmeans_.fit(X)
        return self

    def transform(self, X, y=None) -> Optional[np.ndarray]:
        return self.kmeans_.cluster_centers_ if self.kmeans_ else None


class BestPolisKMeans(BaseEstimator):
    """
    A clusterer that automatically finds optimal k-means clustering using silhouette scores.

    This class provides a scikit-learn-like interface while handling k-selection
    internally using grid search and silhouette scoring.

    Parameters
    ----------
    k_bounds : list of int, default=[2, 5]
        Range of k values to search [min_k, max_k]
    init : {'k-means++', 'random', 'polis'}, default='polis'
        Initialization strategy
    init_centers : array-like, optional
        Initial cluster centers
    random_state : int, optional
        Random state for reproducible results

    Attributes
    ----------
    best_estimator_ : PolisKMeans
        The best fitted estimator
    best_k_ : int
        The optimal number of clusters found
    best_score_ : float
        The best silhouette score achieved
    """

    def __init__(
        self,
        k_bounds: Optional[List[int]] = None,
        init: str = "polis",
        init_centers: Optional[ArrayLike] = None,
        random_state: Optional[int] = None,
    ):
        self.k_bounds = k_bounds or [2, 5]
        self.init = init
        self.init_centers = init_centers
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_k_ = None
        self.best_score_ = None

    def fit(self, X: NDArray) -> 'BestPolisKMeans':
        """Fit the clusterer and find optimal number of clusters using silhouette scores."""
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

        return self

    def fit_predict(self, X: NDArray, y=None, **kwargs) -> Optional[np.ndarray]:
        """Fit the clusterer and return cluster labels."""
        self.fit(X)
        return self.best_estimator_.labels_ if self.best_estimator_ is not None else None
