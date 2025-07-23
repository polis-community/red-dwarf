from typing import TYPE_CHECKING, cast

from reddwarf.exceptions import try_import
from reddwarf.sklearn.cluster import BestPolisKMeans
from .registry import register_clusterer

if TYPE_CHECKING:
    import hdbscan as hdbscan_module


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
    return BestPolisKMeans(**defaults)


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
