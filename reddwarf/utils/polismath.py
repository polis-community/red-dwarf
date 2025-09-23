import numpy as np
from typing import Any, Literal, Tuple

from numpy.typing import NDArray
from reddwarf.types.polis import (
    PolisBaseClusters,
    PolisGroupCluster,
    PolisGroupClusterExpanded,
    GroupId,
    ParticipantId,
)


## POLISMATH HELPERS
#
# These functions are generally used to transform data from the polismath object
# that gets returned from the /api/v3/math/pca2 endpoint.

def expand_group_clusters_with_participants(
    group_clusters: list[PolisGroupCluster],
    base_clusters: PolisBaseClusters,
) -> list[PolisGroupClusterExpanded]:
    """
    Expand group clusters to include direct participant IDs instead of base cluster IDs.

    Args:
        group_clusters (list[PolisGroupCluster]): Group cluster data from Polis math data, with base cluster ID membership.
        base_clusters (PolisBaseClusters): Base cluster data from Polis math data, with participant ID membership.

    Returns:
        group_clusters_with_participant_ids (list[PolisGroupClusterExpanded]): A list of group clusters with participant IDs.
    """
    # Create a mapping from base cluster ID to member participant IDs
    _, base_cluster_to_participant_map = create_bidirectional_id_maps(base_clusters)

    # Create the new group-clusters-participant list
    group_clusters_with_participant_ids = []

    for group in group_clusters:
        # Get all participant IDs for this group by expanding and flattening the base cluster members.
        group_participant_ids = [
            participant_id
            for base_cluster_id in group["members"]
            for participant_id in base_cluster_to_participant_map[base_cluster_id]
        ]

        # Create the new group object with expanded participant IDs
        expanded_group = {
            **group,
            "id": group["id"],
            "members": group_participant_ids,
        }

        group_clusters_with_participant_ids.append(expanded_group)

    return group_clusters_with_participant_ids

def generate_cluster_labels(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> NDArray[np.integer]:
    """
    Transform group_clusters_with_pid into an ordered list of cluster IDs
    sorted by participant ID, suitable for cluster labels.

    Args:
        group_clusters_with_pids (list[PolisGroupClusterExpanded]): List of group clusters with expanded participant IDs

    Returns:
        np.ndarray[GroupId]: A list of cluster IDs ordered by participant ID
    """
    # Create a list of (participant_id, cluster_id) pairs
    participant_cluster_pairs = []

    for cluster in group_clusters_with_pids:
        cluster_id = cluster["id"]
        for participant_id in cluster["members"]:
            participant_cluster_pairs.append((participant_id, cluster_id))

    # Sort the list by participant_id numerically if possible
    try:
        sorted_pairs = sorted(participant_cluster_pairs, key=lambda x: int(x[0]))
    except (ValueError, TypeError):
        sorted_pairs = sorted(participant_cluster_pairs, key=lambda x: x[0])

    # Extract just the cluster IDs in order
    cluster_ids_in_order = [pair[1] for pair in sorted_pairs]

    return np.asarray(cluster_ids_in_order)

def get_all_participant_ids(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> list[ParticipantId]:
    """
    Extract all unique participant IDs from group_clusters_with_pids.

    Args:
        group_clusters_with_pids (list): List of group clusters with expanded participant IDs

    Returns:
        list: A sorted list of all unique participant IDs
    """
    # Gather all participant IDs from all clusters
    all_participant_ids = []

    for cluster in group_clusters_with_pids:
        all_participant_ids.extend(cluster["members"])

    # Remove duplicates and sort
    unique_participant_ids = sorted(set(all_participant_ids))

    return unique_participant_ids

def extract_data_from_polismath(math_data: Any) -> Tuple[list[ParticipantId], NDArray[np.integer]]:
    """
    A helper to extract specific types of data from polismath data, to avoid having to recalculate it.

    This is mostly helpful for debugging and working with the existing Polis platform API.

    Args:
        math_data (Any): The polismath data object that is returned from /api/v3/math/pca2

    Returns:
        list[ParticipantId]: A sorted list of all clustered/grouped Polis participant IDs
        list[GroupId]: A list of all cluster labels (aka group IDs), sorted by participant ID
    """
    group_clusters_with_pids = expand_group_clusters_with_participants(
        group_clusters=math_data["group-clusters"],
        base_clusters=math_data["base-clusters"],
    )

    grouped_participant_ids = get_all_participant_ids(group_clusters_with_pids)
    cluster_labels = generate_cluster_labels(group_clusters_with_pids)

    return grouped_participant_ids, cluster_labels

def create_bidirectional_id_maps(base_clusters):
    """
    Create bidirectional mappings between base_cluster_id and participant_ids.

    Args:
        base_clusters (dict): Dictionary containing 'id' and 'members' lists

    Returns:
        tuple (participant_to_base_cluster, base_cluster_to_participants): maps going in either direction
    """
    participant_to_base_cluster = {
        participant: base_cluster
        for base_cluster, members in zip(base_clusters['id'], base_clusters['members'])
        for participant in members
    }

    base_cluster_to_participants = {
        base_cluster: members
        for base_cluster, members in zip(base_clusters['id'], base_clusters['members'])
    }

    return participant_to_base_cluster, base_cluster_to_participants

def get_corrected_centroid_guesses(
    polis_math_data: dict,
    source: Literal["group-clusters", "base-clusters"] = "group-clusters",
    flip_x: bool = True,
    flip_y: bool = True,
):
    """
    A helper to extract and correct centroid guesses from polismath data.

    This is helpful to seed new rounds of KMeans locally.

    NOTE: It seems that there's an inversion somewhere in the polis codebase, and so
    we usually (but not always) need to invert their coordinates in relation to ours.

    Given the consistency of this intervention, it's likely not a PCA artifact,
    and instead relates to using inverting the sign of agree/disagree in
    varioius places in the Polis codebase.

    This function supports multiple sources of centroid guesses:
    - "group-clusters": Uses the `center` field of each group
    - "base-clusters": Uses the parallel `x` and `y` fields to reconstruct [x, y] coords

    Arguments:
        polis_math_data (dict): The polismath data from the Polis API
        source (str): Where to extract centroid data from. One of "group-clusters" or "base-clusters".
        flip_x (bool): Whether to correctively flip X sign (usually helpful)
        flip_y (bool): Whether to correctively flip Y sign (usually helpful)

    Returns:
        centroids: A list of centroid [x,y] coord guesses
    """
    if source == "group-clusters":
        extracted_centroids = [group["center"] for group in polis_math_data["group-clusters"]]
    elif source == "base-clusters":
        base = polis_math_data["base-clusters"]
        x_vals = base.get("x", [])
        y_vals = base.get("y", [])

        if len(x_vals) != len(y_vals):
            raise ValueError("Mismatched lengths between 'x' and 'y' values in base-clusters")

        extracted_centroids = [[x, y] for x, y in zip(x_vals, y_vals)]
    else:
        # Defensive fallback in case `source` comes from dynamic or untyped input
        raise ValueError(f"Unknown source '{source}'. Must be 'group-clusters' or 'base-clusters'.")

    corrected_centroids = [
        [
            -xy[0] if flip_x else xy[0],
            -xy[1] if flip_y else xy[1],
        ] for xy in extracted_centroids
    ]

    return corrected_centroids