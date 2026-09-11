import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.implementations.base import AnalysisOutcome
from reddwarf.data_loader import Loader
from reddwarf.utils.statements import process_statements
from reddwarf.utils.polismath import extract_data_from_polismath
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
from pandas._testing import assert_dict_equal
from tests import helpers


# This test will only match polismath for sub-100 participant convos.
# TODO: Investigate why this small convo fails: "r5apwkphpuxxhf9wnfemj"
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_run_clustering_real_data_small(polis_convo_data):
    fixture = polis_convo_data
    math_data = helpers.flip_signs_by_key(
        nested_dict=fixture.math_data,
        keys=[
            "pca.center",
            "pca.comment-projection",
            "base-clusters.x",
            "base-clusters.y",
            "group-clusters[*].center",
        ],
    )

    # We force kmeans to find a specific value because Polis does this for something they call k-smoothing.
    # TODO: Get k-smoothing working, and simulate how k is held back.
    force_group_count = len(math_data["group-clusters"])

    # Transpose base cluster coords into participant_ids
    expected_projected_ptpts = helpers.transform_base_clusters_to_participant_coords(
        math_data["base-clusters"]
    )

    max_group_count = 5
    init_centers = [group["center"] for group in math_data["group-clusters"]]

    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            # Loading these helps generate mod_out_statement_ids
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )

    _, _, mod_out_statement_ids, meta_statement_ids = process_statements(
        statement_data=loader.comments_data
    )

    result = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
        keep_participant_ids=fixture.keep_participant_ids,
        max_group_count=max_group_count,
        init_centers=init_centers,
        force_group_count=force_group_count,
    )
    assert result.outcome == AnalysisOutcome.SUCCESS
    result = result.result

    # Check group-aware-consensus calculations.
    calculated = helpers.simulate_api_response(
        result.statements_df["group-aware-consensus"].items()
    )
    expected = math_data["group-aware-consensus"]
    assert_dict_equal(calculated, expected)

    # Check PCA components and means
    assert pytest.approx(result.reducer.components_[0]) == math_data["pca"]["comps"][0]
    assert pytest.approx(result.reducer.components_[1]) == math_data["pca"]["comps"][1]
    assert pytest.approx(result.reducer.mean_) == math_data["pca"]["center"]

    # Check projected statements
    # Convert [[x_1, y_1], [x_2, y_2], ...] into [[x_1, x_2, ...], list[y_1, y_2, ...]]
    actual = result.statements_df[["x", "y"]].values.transpose()
    expected = math_data["pca"]["comment-projection"]
    assert_array_almost_equal(actual, expected)

    # Check projected participants
    # Ensure we have as many expected coords as calculated coords.
    clustered_participants_df = result.participants_df.loc[result.participants_df["to_cluster"], :]
    assert len(clustered_participants_df.index) == len(expected_projected_ptpts)

    for projection in expected_projected_ptpts:
        expected_xy = projection["xy"]
        calculated_xy = result.participants_df.loc[
            projection["participant_id"], ["x", "y"]
        ].values

        assert calculated_xy == pytest.approx(expected_xy)

    # Check that the cluster labels all match when K is forced to match.
    _, expected_cluster_labels = extract_data_from_polismath(math_data)
    clustered_participants_df = result.participants_df.loc[result.participants_df["to_cluster"], :]
    calculated_cluster_labels = clustered_participants_df["cluster_id"].values
    assert_array_equal(calculated_cluster_labels, expected_cluster_labels)  # type:ignore

    # Check extremity calculation
    expected = math_data["pca"]["comment-extremity"]
    calculated = result.statements_df["extremity"].tolist()
    assert_array_almost_equal(expected, calculated)

    # Check comment-priority calculcation
    expected = math_data["comment-priorities"]

    calculated = helpers.simulate_api_response(result.statements_df["priority"].items())
    assert_dict_equal(expected, calculated)

    # Check representative statements
    expected = math_data["repness"]
    calculated = helpers.simulate_api_response(result.repness)
    assert_dict_equal(expected, calculated)

    # Check consensus statements
    expected = math_data["consensus"]
    calculated = result.consensus
    assert_dict_equal(expected, calculated)


# For testing our code agaisnt real data with against larger conversations,
# we'll need to implement base clustering.
@pytest.mark.skip
@pytest.mark.parametrize(
    "polis_convo_data", ["medium-no-meta", "medium-with-meta"], indirect=True
)
def test_run_clustering_real_data_medium(polis_convo_data):
    raise NotImplementedError


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_clustering_is_reproducible(polis_convo_data):
    fixture = polis_convo_data

    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            # Loading these helps generate mod_out_statement_ids
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )

    _, _, mod_out_statement_ids, _ = process_statements(
        statement_data=loader.comments_data
    )

    cluster_run_1 = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    assert cluster_run_1.outcome == AnalysisOutcome.SUCCESS
    cluster_run_1 = cluster_run_1.result

    centers_1 = cluster_run_1.clusterer.cluster_centers_ if cluster_run_1.clusterer else None

    cluster_run_2 = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        init_centers=centers_1,
    )
    assert cluster_run_2.outcome == AnalysisOutcome.SUCCESS
    cluster_run_2 = cluster_run_2.result

    # same number of clusters
    centers_1 = cluster_run_1.clusterer.cluster_centers_ if cluster_run_1.clusterer else []
    centers_2 = cluster_run_2.clusterer.cluster_centers_ if cluster_run_2.clusterer else []
    assert len(centers_1) == len(centers_2)
    assert_array_equal(centers_1, centers_2)

    # projected statements and cluster IDs
    assert_frame_equal(
        cluster_run_1.participants_df.loc[
            cluster_run_1.participants_df["to_cluster"],
            ["x", "y", "cluster_id"],
        ],
        cluster_run_2.participants_df.loc[
            cluster_run_2.participants_df["to_cluster"],
            ["x", "y", "cluster_id"],
        ]
    )
    # components/eigenvectors
    assert (
        cluster_run_1.reducer.components_.tolist() == cluster_run_2.reducer.components_.tolist()
    )
    # statement means
    assert cluster_run_1.reducer.mean_.tolist() == cluster_run_2.reducer.mean_.tolist()
