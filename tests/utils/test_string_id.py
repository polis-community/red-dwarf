import pytest
import numpy as np
from tests import helpers
from tests.fixtures import polis_convo_data
from numpy.testing import assert_array_equal, assert_array_almost_equal
from reddwarf.utils import stats, polismath, matrix, statements as stmnts
from pandas.testing import assert_frame_equal
from pandas._testing import assert_dict_equal
from reddwarf.types.polis import PolisRepness
from reddwarf.data_loader import Loader
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.implementations.base import run_pipeline
from reddwarf.implementations.polis import run_clustering
from reddwarf.utils.statements import process_statements
from reddwarf.utils.polismath import extract_data_from_polismath
from tests.helpers import get_grouped_statement_ids


# This test is to ensure that the run_pipeline function can handle string IDs for statements and participants
@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_basic_pipeline_execution_with_string_ids(reducer,polis_convo_data):
    """Test that run_pipeline supports string statement and participant id's and outputs expected data"""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Preprocess votes to convert participant and statement IDs to strings
    preprocessed_vote_data = helpers.convert_ids_to_strings(votes)

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

    print(f"preprocessed_mathdata_results:{preprocessed_mathdata_results}\n")
    assert preprocessed_vote_matrix == preprocessed_mathdata_results



# This test is the longest running path  to ensure that underline functions utilizing statement_id can handle string IDs for statements and participants
# This test is a copy of test_run_clustering_real_data_small, for comments and context see tests/implementations/test_polis.py function test_run_clustering_real_data_small()
@pytest.mark.parametrize(
    "polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True
)
def test_run_pipeline_with_string_statement_ids(polis_convo_data):
    """Longest path to test that run_clustering supports string statement and participant id's and outputs expected data"""
    fixture = polis_convo_data

    flipped_math_data = helpers.flip_signs_by_key(
        nested_dict=fixture.math_data,
        keys=[
            "pca.center",
            "pca.comment-projection",
            "base-clusters.x",
            "base-clusters.y",
            "group-clusters[*].center",
        ],
    )

    math_data = {str(k):v for k,v in flipped_math_data.items()}

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

    processed_votes = helpers.convert_ids_to_strings(loader.votes_data)
    processed_comments = helpers.convert_ids_to_strings(loader.comments_data)

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

    processed_expected_projected_ptpts = helpers.convert_ids_to_strings(expected_projected_ptpts)

    for projection in processed_expected_projected_ptpts:
        # print(f"Checking projection for participant {projection['participant_id']}")
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
    

# This test is as copy of test_run_clustering_is_reproducible
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_clustering_reproducibility_with_string_ids(polis_convo_data):
    """Test that run_clustering supports string statement and participant id's and outputs expected data"""
    fixture = polis_convo_data

    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )

    _, _, mod_out_statement_ids, _ = process_statements(
        statement_data=helpers.convert_ids_to_strings(loader.comments_data),
    )

    cluster_run_1 = run_clustering(
        votes=helpers.convert_ids_to_strings(loader.votes_data),
        mod_out_statement_ids=mod_out_statement_ids,
    )

    centers_1 = cluster_run_1.clusterer.cluster_centers_ if cluster_run_1.clusterer else None

    cluster_run_2 = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        init_centers=centers_1,
    )

    centers_1 = cluster_run_1.clusterer.cluster_centers_ if cluster_run_1.clusterer else []
    centers_2 = cluster_run_2.clusterer.cluster_centers_ if cluster_run_2.clusterer else []
    assert len(centers_1) == len(centers_2)
    assert_array_equal(centers_1, centers_2)

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
    assert (
        cluster_run_1.reducer.components_.tolist() == cluster_run_2.reducer.components_.tolist()
    )
    assert cluster_run_1.reducer.mean_.tolist() == cluster_run_2.reducer.mean_.tolist()


# This test is a copy of test_select_representative_statements_real_data
@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"],
    indirect=True,
)
def test_representative_statement_selection_with_string_ids(polis_convo_data):
    fixture = polis_convo_data
    grouped_stats_df, _ = setup_test_str(fixture)

    fixture_math_data_mod_out = list(map(str, fixture.math_data["mod-out"]))

    polis_repness = stats.select_representative_statements(
        grouped_stats_df=grouped_stats_df,
        mod_out_statement_ids=fixture_math_data_mod_out,
        pick_max=5,
    )

    calculated_repness: PolisRepness = polis_repness
    expected_repness: PolisRepness = fixture.math_data["repness"]
    
    # Compare the selected statements calculated from those generated by polismath.
    calculated_grounded_statement_ids = get_grouped_statement_ids(calculated_repness)
    expected_grounded_statement_ids = get_grouped_statement_ids(expected_repness)

    assert  calculated_grounded_statement_ids == expected_grounded_statement_ids

# This test is a copy of test_calculate_comment_statistics_dataframes_grouped_stats_df_real_data
@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"],
    indirect=True,
)
def test_calculate_comment_statistics_dataframes_grouped_stats_df_real_data(
    polis_convo_data,
):
    fixture = polis_convo_data
    grouped_stats_df, _ = setup_test_str(fixture)

    # Cycle through all the expected data calculated by Polis platform
    for group_id, statements in fixture.math_data["repness"].items():
        group_id = int(group_id)
        for st in statements:
            expected_repr = st["repness"]
            expected_repr_test = st["repness-test"]
            expected_prob = st["p-success"]
            expected_prob_test = st["p-test"]

            # Fetch matching calculated values for comparison.
            keys = ["prob", "prob_test", "repness", "repness_test"]
            if st["repful-for"] == "agree":
                key_map = dict(zip(keys, ["pa", "pat", "ra", "rat"]))
            else:  # disagree
                key_map = dict(zip(keys, ["pd", "pdt", "rd", "rdt"]))

            actual = {
                k: grouped_stats_df.loc[(group_id, str(st["tid"])), v]
                for k, v in key_map.items()
            }



            assert actual["prob"] == pytest.approx(expected_prob)
            assert actual["prob_test"] == pytest.approx(expected_prob_test)
            assert actual["repness"] == pytest.approx(expected_repr)
            assert actual["repness_test"] == pytest.approx(expected_repr_test)

# Helper function to process participant IDs used in test_representative_statement_selection_with_string_ids
def setup_test_str(fixture):
    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )
    VOTES = loader.votes_data

    processed_votes = helpers.convert_ids_to_strings(VOTES)    

    raw_vote_matrix = matrix.generate_raw_matrix(votes=processed_votes)

    # preprend "p" str to index in order to match converted string id
    raw_vote_matrix.index = raw_vote_matrix.index.map("p{}".format) 
   
    all_clustered_participant_ids, cluster_labels = (
        polismath.extract_data_from_polismath(fixture.math_data)
    )

    processed_participant_ids = helpers.convert_participant_ids_to_strings(
        all_clustered_participant_ids
    )

    # Generate stats all groups and all statements.
    grouped_stats_df, gac_df = stats.calculate_comment_statistics_dataframes(
        vote_matrix=raw_vote_matrix.loc[processed_participant_ids, :],
        cluster_labels=cluster_labels,
    )
    return grouped_stats_df, gac_df
