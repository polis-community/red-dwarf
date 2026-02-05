from numpy.typing import NDArray
import pandas as pd
import numpy as np
from reddwarf.sklearn.model_selection import GridSearchNonCV
from reddwarf.sklearn.cluster import PolisKMeans
from sklearn.metrics import silhouette_score
from typing import List, Optional

RangeLike = int | tuple[int, int] | list[int]

def to_range(r: RangeLike) -> range:
    """
    Creates an inclusive range from a list, tuple, or int.

    Examples:
        ```
        to_range(2) # [2, 3]
        to_range([2, 5]) # [2, 3, 4, 5]
        to_range((2, 5)) # [2, 3, 4, 5]
        ```
    """
    if isinstance(r, int):
        start = end = r
    elif isinstance(r, (tuple, list)) and len(r) == 2:
        start, end = r
    else:
        raise ValueError("Expected int or a 2-element tuple/list")

    return range(start, end + 1)  # inclusive

# TODO: Start passing init_centers based on /math/pca2 endpoint data,
# and see how often we get the same clusters.
def run_kmeans(
        dataframe: pd.DataFrame,
        n_clusters: int = 2,
        init="k-means++",
        # TODO: Improve this type. 3d?
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
) -> PolisKMeans:
    """
    Runs K-Means clustering on a 2D DataFrame of xy points, for a specific K,
    and returns labels for each row and cluster centers. Optionally accepts
    guesses on cluster centers, and a random_state to reproducibility.

    Args:
        dataframe (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        n_clusters (int): How many clusters k to assume.
        init (string): The cluster initialization strategy. See `PolisKMeans` docs.
        init_centers (List): A list of xy coordinates to use as initial center guesses. See `PolisKMeans` docs.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        kmeans (PolisKMeans): The estimator object returned from PolisKMeans.
    """
    # TODO: Set random_state to a value eventually, so calculation is deterministic.
    kmeans = PolisKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init=init,
        init_centers=init_centers,
    ).fit(dataframe)

    return kmeans

def find_best_kmeans(
        X_to_cluster: NDArray,
        k_bounds: RangeLike = [2, 5],
        init="k-means++",
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
) -> tuple[int, float, PolisKMeans | None]:
    """
    Use silhouette scores to find the best number of clusters k to assume to fit the data.

    Args:
        X_to_cluster (NDArray): A n-D numpy array.
        k_bounds (RangeLike): An upper and low bound on n_clusters to test for. (Default: [2, 5])
        init_centers (List): A list of xy coordinates to use as initial center guesses.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        best_k (int): Ideal number of clusters.
        best_silhouette_score (float): Silhouette score for this K value.
        best_kmeans (PolisKMeans | None): The optimal fitted estimator returned from PolisKMeans.
    """
    param_grid = {
        "n_clusters": to_range(k_bounds),
    }

    def scoring_function(estimator, X):
        labels = estimator.fit_predict(X)
        return silhouette_score(X, labels)

    search = GridSearchNonCV(
        param_grid=param_grid,
        scoring=scoring_function,
        estimator=PolisKMeans(
            init=init, # strategy
            init_centers=init_centers, # guesses
            random_state=random_state,
        ),
    )

    search.fit(X_to_cluster)

    best_k = search.best_params_['n_clusters']
    best_silhouette_score = search.best_score_
    best_kmeans = search.best_estimator_

    return best_k, best_silhouette_score, best_kmeans