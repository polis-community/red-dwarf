import pytest
import numpy as np
from reddwarf.utils.clusterer.kmeans import (
    calculate_kmeans_silhouette_score,
    run_kmeans,
    find_best_kmeans,
)
from tests.fixtures import polis_convo_data
from tests.helpers import transform_base_clusters_to_participant_coords
import pandas as pd


def test_find_best_kmeans_rejects_singleton_clusters():
    """find_best_kmeans should never select a k that produces a singleton cluster."""
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    outlier = np.array([[10, 10]])
    X = np.vstack([cluster1, cluster2, outlier])

    best_k, _, best_kmeans = find_best_kmeans(
        X_to_cluster=X,
        k_bounds=[2, 5],
        random_state=42,
    )

    if best_kmeans is not None:
        unique, counts = np.unique(best_kmeans.labels_, return_counts=True)
        assert counts.min() >= 2, (
            f"k={best_k} produced singleton cluster(s): {dict(zip(unique, counts))}"
        )


def test_calculate_kmeans_silhouette_score_returns_none_when_undefined():
    X = np.array([[0, 0], [1, 1]])
    labels = np.array([0, 0])

    assert calculate_kmeans_silhouette_score(X_to_cluster=X, labels=labels) is None


def test_calculate_kmeans_silhouette_score_discourages_singleton_clusters():
    X = np.array([[0, 0], [0, 1], [10, 10]])
    labels = np.array([0, 0, 1])

    assert calculate_kmeans_silhouette_score(X_to_cluster=X, labels=labels) == -1.0

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_kmeans_real_data_reproducible(polis_convo_data):
    fixture = polis_convo_data

    expected_cluster_centers = [group["center"] for group in fixture.math_data["group-clusters"]]
    cluster_count = len(expected_cluster_centers)

    projected_participants = transform_base_clusters_to_participant_coords(fixture.math_data["base-clusters"])
    projected_participants_df = pd.DataFrame([
        {
            "participant_id": item["participant_id"],
            "x": item["xy"][0],
            "y": item["xy"][1],
        }
        for item in projected_participants
    ]).set_index("participant_id")

    calculated_kmeans = run_kmeans(
        dataframe=projected_participants_df,
        init_centers=expected_cluster_centers,
        n_clusters=cluster_count,
    )

    # Ensure same number of clusters
    assert len(calculated_kmeans.cluster_centers_) == len(expected_cluster_centers)

    # Ensure each is the same
    for i, _ in enumerate(calculated_kmeans.cluster_centers_):
        assert calculated_kmeans.cluster_centers_.tolist()[i] == pytest.approx(expected_cluster_centers[i])

# NOTE: "small-no-meta" fixture doesn't work because wants to find 4 clusters, whereas real data from polismath says 3.
# This is likely due to k-smoothing holding back the k value at 3 in polismath, and we're finding the real current one.
@pytest.mark.parametrize("polis_convo_data", ["small-with-meta"], indirect=True)
def test_find_best_kmeans_real_data(polis_convo_data):
    fixture = polis_convo_data
    MAX_GROUP_COUNT = 5

    # Get centers from polismath.
    expected_centers = [group["center"] for group in fixture.math_data["group-clusters"]]

    projected_participants = transform_base_clusters_to_participant_coords(fixture.math_data["base-clusters"])
    projected_participants_df = pd.DataFrame([
        {
            "participant_id": item["participant_id"],
            "x": item["xy"][0],
            "y": item["xy"][1],
        }
        for item in projected_participants
    ]).set_index("participant_id")

    results = find_best_kmeans(
        X_to_cluster=projected_participants_df,
        k_bounds=[2, MAX_GROUP_COUNT],
        # Pad center guesses to have enough values for testing up to max k groups.
        init_centers=expected_centers
    )
    _, _, optimal_kmeans = results # for documentation

    calculated_centers = optimal_kmeans.cluster_centers_.tolist() if optimal_kmeans else []

    assert len(expected_centers) == len(calculated_centers)
    for i, _ in enumerate(expected_centers):
        assert expected_centers[i] == pytest.approx(calculated_centers[i])
