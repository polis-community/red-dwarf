import pytest
from reddwarf.sklearn.cluster import BestPolisKMeans, PolisKMeans
from tests.fixtures import polis_convo_data
from tests.helpers import transform_base_clusters_to_participant_coords
import pandas as pd

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_polis_kmeans_real_data_reproducible(polis_convo_data):
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

    calculated_kmeans = PolisKMeans(
        n_clusters=cluster_count,
        init="k-means++",
        init_centers=expected_cluster_centers,
    ).fit(projected_participants_df)

    # Ensure same number of clusters
    assert len(calculated_kmeans.cluster_centers_) == len(expected_cluster_centers)

    # Ensure each is the same
    for i, _ in enumerate(calculated_kmeans.cluster_centers_):
        assert calculated_kmeans.cluster_centers_.tolist()[i] == pytest.approx(expected_cluster_centers[i])

# NOTE: "small-no-meta" fixture doesn't work because wants to find 4 clusters, whereas real data from polismath says 3.
# This is likely due to k-smoothing holding back the k value at 3 in polismath, and we're finding the real current one.
@pytest.mark.parametrize("polis_convo_data", ["small-with-meta"], indirect=True)
def test_best_polis_kmeans_real_data(polis_convo_data):
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

    from reddwarf.utils.clusterer.base import run_clusterer

    # Test using run_clusterer (which is what the pipeline actually uses)
    clusterer_result = run_clusterer(
        clusterer="kmeans",
        X_participants_clusterable=projected_participants_df.values,
        k_bounds=[2, MAX_GROUP_COUNT],
        init_centers=expected_centers
    )

    cluster_centers = getattr(clusterer_result, 'cluster_centers_', None)
    calculated_centers = cluster_centers.tolist() if cluster_centers is not None else []

    # Verify init_centers_used_ attribute is available (this was the original bug)
    assert hasattr(clusterer_result, 'init_centers_used_')
    init_centers_used = getattr(clusterer_result, 'init_centers_used_', None)
    assert init_centers_used is not None
    assert init_centers_used.shape[0] == len(calculated_centers)

    assert len(expected_centers) == len(calculated_centers)
    for i, _ in enumerate(expected_centers):
        assert expected_centers[i] == pytest.approx(calculated_centers[i])
