from typing import List, Optional
import pandas as pd
from reddwarf.sklearn.cluster import PolisKMeans


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