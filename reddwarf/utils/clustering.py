import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Optional


# TODO: Start passing init_centers based on /math/pca2 endpoint data,
# and see how often we get the same clusters.
def run_kmeans(
        dataframe: pd.DataFrame,
        n_clusters: int = 2,
        # TODO: Improve this type. 3d?
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
) -> Tuple[np.ndarray | None, np.ndarray]:
    """
    Runs K-Means clustering on a 2D DataFrame of xy points, for a specific K,
    and returns labels for each row and cluster centers. Optionally accepts
    guesses on cluster centers, and a random_state to reproducibility.

    Args:
        dataframe (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        n_clusters (int): How many clusters k to assume.
        init_centers (List): A list of xy coordinates to use as initial center guesses.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        cluster_labels (np.ndarray | None): A list of zero-indexed labels for each row in the dataframe
        cluster_centers (np.ndarray): A list of center coords for clusters.
    """
    if init_centers:
        # Pass an array of xy coords to see kmeans guesses.
        init_arg = init_centers[:n_clusters]
    else:
        # Use the default strategy in sklearn.
        init_arg = "k-means++"
    # TODO: Set random_state to a value eventually, so calculation is deterministic.
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init = init_arg, # type:ignore (because sklearn things it's just a string)
        n_init="auto"
    ).fit(dataframe)

    return kmeans.labels_, kmeans.cluster_centers_

def find_optimal_k(
        projected_data: pd.DataFrame,
        max_group_count: int = 5,
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
        debug: bool = False,
) -> Tuple[int, float, np.ndarray | None, np.ndarray | None]:
    """
    Use silhouette scores to find the best number of clusters k to assume to fit the data.

    Args:
        projected_data (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        max_group_count (int): The max K number of groups to test for. (Default: 5)
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        debug (bool): Whether to print debug output. (Default: False)

    Returns:
        optimal_k (int): Ideal number of clusters.
        optimal_silhouette_score (float): Silhouette score for this K value.
        optimal_cluster_labels (np.ndarray | None): A list of index labels assigned a group to each row in projected_date.
        optimal_cluster_centers (np.ndarray | None): A list of xy centers for the optimal clusters.
    """
    K_RANGE = range(2, max_group_count+1)
    k_best = 0 # Best K so far.
    best_silhouette_score = -np.inf
    best_cluster_labels = None
    best_cluster_centers = None

    for k_test in K_RANGE:
        cluster_labels, cluster_centers = run_kmeans(
            dataframe=projected_data,
            n_clusters=k_test,
            init_centers=init_centers,
            random_state=random_state,
        )
        this_silhouette_score = silhouette_score(projected_data, cluster_labels)
        if debug:
            print(f"{k_test=}, {this_silhouette_score=}")
        if this_silhouette_score >= best_silhouette_score:
            k_best = k_test
            best_silhouette_score = this_silhouette_score
            best_cluster_labels = cluster_labels
            best_cluster_centers = cluster_centers

    optimal_k = k_best
    optimal_silhouette = best_silhouette_score
    optimal_cluster_labels = best_cluster_labels
    optimal_cluster_centers = best_cluster_centers

    return optimal_k, optimal_silhouette, optimal_cluster_labels, optimal_cluster_centers