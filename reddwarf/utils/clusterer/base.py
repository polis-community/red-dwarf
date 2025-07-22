from typing import Optional, Union, TypeAlias, TYPE_CHECKING, Any
from numpy.typing import NDArray
from reddwarf.utils.clusterer.registry import get_clusterer
from reddwarf.utils.clusterer import load_builtins

if TYPE_CHECKING:
    from hdbscan import HDBSCAN
    from reddwarf.sklearn.cluster import PolisKMeans

ClustererModel: TypeAlias = Union["HDBSCAN", "PolisKMeans", Any]

# Load builtin clusterers
load_builtins()


def run_clusterer(
    clusterer: str,
    X_participants_clusterable: NDArray,
    force_group_count: Optional[int] = None,
    max_group_count: int = 5,
    init_centers: Optional[list] = None,
    random_state: Optional[int] = None,
    **clusterer_kwargs,
) -> Optional[ClustererModel]:
    """
    Run a clusterer on participant data using the registry system.

    Args:
        clusterer: Name of the registered clusterer to use
        X_participants_clusterable: Array of participant coordinates to cluster
        force_group_count: Force a specific number of clusters (for k-means)
        max_group_count: Maximum number of clusters to test (for k-means)
        init_centers: Initial cluster center coordinates
        random_state: Random state for reproducibility
        **clusterer_kwargs: Additional parameters to pass to the clusterer

    Returns:
        Fitted clusterer model or None if clustering fails
    """
    # Handle k-means specific parameters
    if clusterer == "kmeans":
        if force_group_count:
            k_bounds = [force_group_count, force_group_count]
        else:
            k_bounds = [2, max_group_count]

        clusterer_kwargs.update({
            'k_bounds': k_bounds,
            'init_centers': init_centers,
            'random_state': random_state,
        })

    # Use the registry system for all clusterers
    try:
        clusterer_instance = get_clusterer(clusterer, **clusterer_kwargs)

        # Fit the clusterer
        if hasattr(clusterer_instance, 'fit'):
            clusterer_instance.fit(X_participants_clusterable)

        return clusterer_instance

    except ValueError as e:
        raise NotImplementedError(f"Clusterer '{clusterer}' not registered: {e}")
