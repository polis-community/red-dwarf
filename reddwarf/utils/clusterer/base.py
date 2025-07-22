from typing import Literal, Optional, Union, TypeAlias, TYPE_CHECKING
from numpy.typing import NDArray
from reddwarf.exceptions import try_import
from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.utils.clusterer.kmeans import find_best_kmeans

if TYPE_CHECKING:
    from hdbscan import HDBSCAN
    from reddwarf.sklearn.cluster import PolisKMeans

ClustererModel: TypeAlias = Union["HDBSCAN", "PolisKMeans"]
ClustererType: TypeAlias = Literal["hdbscan", "kmeans"]


def run_clusterer(
    X_participants_clusterable: NDArray,
    clusterer: ClustererType = "kmeans",
    force_group_count=None,
    max_group_count=5,
    init_centers=None,
    random_state=None,
    **clusterer_kwargs,
) -> Optional[ClustererModel]:
    match clusterer:
        case "kmeans":
            if force_group_count:
                k_bounds = [force_group_count, force_group_count]
            else:
                k_bounds = [2, max_group_count]

            _, _, kmeans = find_best_kmeans(
                X_to_cluster=X_participants_clusterable,
                k_bounds=k_bounds,
                # Force polis strategy of initiating cluster centers. See: PolisKMeans.
                init="polis",
                init_centers=init_centers,
                random_state=random_state,
                # TODO: Support passing in arbitrary clusterer_kwargs.
            )

            return kmeans

        case "hdbscan":
            hdbscan = try_import("hdbscan", extra="alt-algos")

            hdb = hdbscan.HDBSCAN(**clusterer_kwargs)
            hdb.fit(X_participants_clusterable)

            return hdb
        case _:
            raise NotImplementedError("clusterer type unknown")
