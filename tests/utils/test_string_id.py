import pytest
import numpy as np
from tests import helpers
from tests.fixtures import polis_convo_data
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
from pandas._testing import assert_dict_equal
from reddwarf.data_loader import Loader
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.implementations.base import run_pipeline
from reddwarf.implementations.polis import run_clustering
from reddwarf.utils.statements import process_statements
from reddwarf.utils.polismath import extract_data_from_polismath


# This test is to ensure that the run_pipeline function can handle string IDs for statements and participants
@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_run_pipeline_with_statement_and_participant_id_casting(reducer,polis_convo_data):
    """Test that run_pipeline supports string statement and participant id's and outputs expected data"""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Preprocess votes to convert participant and statement IDs to strings
    preprocessed_vote_data = helpers.preprocess_data(votes)

    # preprocessed clustering run_pipeine
    preprocessed_clustering_result = run_pipeline(
       votes = preprocessed_vote_data,
       reducer=reducer
    )

    # Get preprocessed vote dict
    preprocessed_vote_matrix = preprocessed_clustering_result.raw_vote_matrix.count(axis="columns").to_dict()

    # Expected values from preprocessed fixture
    preprocessed_mathdata_results = fixture.math_data["user-vote-counts"]
    preprocessed_mathdata_results = {str(k):v for k,v in preprocessed_mathdata_results.items()}

    assert preprocessed_vote_matrix == preprocessed_mathdata_results



# This test is the longest running path  to ensure that underline functions utilizing statement_id can handle string IDs for statements and participants
# This test is a copy of test_run_clustering_real_data_small, for comments and context see tests/implementations/test_polis.py function test_run_clustering_real_data_small()
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_run_pipeline_with_string_statement_ids(polis_convo_data):
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

    force_group_count = len(math_data["group-clusters"])

    expected_projected_ptpts = helpers.transform_base_clusters_to_participant_coords(
        math_data["base-clusters"]
    )

    max_group_count = 5
    init_centers = [group["center"] for group in math_data["group-clusters"]]

    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )

    processed_votes = helpers.preprocess_data(loader.votes_data)
    processed_comments = helpers.preprocess_data(loader.comments_data)

    _, _, mod_out_statement_ids, meta_statement_ids = process_statements(
        statement_data=processed_comments,
    )

    result = run_clustering(
        votes=processed_votes,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
        keep_participant_ids=[str(pid) for pid in fixture.keep_participant_ids],
        max_group_count=max_group_count,
        init_centers=init_centers,
        force_group_count=force_group_count,
    )
    
    calculated = helpers.simulate_api_response(
        result.statements_df["group-aware-consensus"].items()
    )
    expected = math_data["group-aware-consensus"]
    assert_dict_equal(calculated, expected)

    assert pytest.approx(result.reducer.components_[0]) == math_data["pca"]["comps"][0]
    assert pytest.approx(result.reducer.components_[1]) == math_data["pca"]["comps"][1]
    assert pytest.approx(result.reducer.mean_) == math_data["pca"]["center"]

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
    
