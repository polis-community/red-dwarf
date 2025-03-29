import numpy as np
from typing import Any, Tuple

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

    # Sort the list by participant_id
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