import pytest
import numpy as np
from pandas.testing import assert_frame_equal
from reddwarf.implementations.base import (
    AnalysisOutcome,
    InsufficientDataReason,
    calculate_projection_silhouette_score,
    prepare_pca_projection,
    run_kmeans_on_pca_projection,
    run_pipeline,
)
from reddwarf.data_loader import Loader
from reddwarf.utils.statements import process_statements
from tests.fixtures import polis_convo_data, convert_ids
from tests import helpers


## Determinism via random_seed for run_pipeline()


@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_deterministic_with_random_state(reducer, polis_convo_data, convert_ids):
    """Test that setting random_state results in the same participant_projections twice in a row."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes =  convert_ids(loader.votes_data)


    random_state = 42

    # Run pipeline twice with same random_state
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=random_state)
    assert result_1.outcome == AnalysisOutcome.SUCCESS
    result_1 = result_1.result

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=random_state)
    assert result_2.outcome == AnalysisOutcome.SUCCESS
    result_2 = result_2.result

    # Results should be identical when using same random_state
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array(
        [result_1.participant_projections[pid] for pid in participant_ids]
    )
    projections_2 = np.array(
        [result_2.participant_projections[pid] for pid in participant_ids]
    )

    np.testing.assert_array_equal(
        projections_1,
        projections_2,
        err_msg=f"{reducer} with random_state should produce identical participant projections",
    )


@pytest.mark.parametrize("reducer", ["pacmap", "localmap"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_not_deterministic_without_random_state(reducer, polis_convo_data, convert_ids):
    """Test that not setting random_state results in different participant_projections across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = convert_ids(loader.votes_data)

    # Run pipeline twice without random_state (should be non-deterministic)
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=None)
    assert result_1.outcome == AnalysisOutcome.SUCCESS
    result_1 = result_1.result

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=None)
    assert result_2.outcome == AnalysisOutcome.SUCCESS
    result_2 = result_2.result

    # Results should be different when not using random_state
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array(
        [result_1.participant_projections[pid] for pid in participant_ids]
    )
    projections_2 = np.array(
        [result_2.participant_projections[pid] for pid in participant_ids]
    )

    # We use assert_raises to expect that arrays are NOT equal
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            projections_1,
            projections_2,
            err_msg=f"{reducer} without random_state should produce different participant projections",
        )


# Note: PCA is always deterministic regardless of random_state
@pytest.mark.parametrize("reducer", ["pca"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_pca_still_deterministic_without_random_state(
    reducer, polis_convo_data, convert_ids
):
    """Test that not setting random_state for PCA is still deterministic across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = convert_ids(loader.votes_data)

    # Run pipeline twice without random_state (PCA should still be deterministic)
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=None)
    assert result_1.outcome == AnalysisOutcome.SUCCESS
    result_1 = result_1.result

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=None)
    assert result_2.outcome == AnalysisOutcome.SUCCESS
    result_2 = result_2.result

    # Results should be identical even without random_state for PCA
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array(
        [result_1.participant_projections[pid] for pid in participant_ids]
    )
    projections_2 = np.array(
        [result_2.participant_projections[pid] for pid in participant_ids]
    )

    np.testing.assert_array_equal(
        projections_1,
        projections_2,
        err_msg=f"{reducer} should produce identical participant projections even without random_state",
    )


## Test for keep_participant_ids with non-existent IDs


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_handles_nonexistent_keep_participant_ids(polis_convo_data):
    """Test that run_pipeline doesn't crash when keep_participant_ids contains IDs that don't exist in vote matrix."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Get actual participant IDs from the votes data
    actual_participant_ids = set(vote["participant_id"] for vote in votes)
    max_existing_id = max(actual_participant_ids)

    # Create a list that includes both existing and non-existent participant IDs
    keep_participant_ids = [
        max_existing_id,  # This ID exists
        max_existing_id + 1000,  # This ID doesn't exist
        max_existing_id + 2000,  # This ID doesn't exist
    ]

    # This should not crash - the bugfix ensures non-existent IDs are filtered out
    result = run_pipeline(
        votes=votes, keep_participant_ids=keep_participant_ids, random_state=42
    )
    assert result.outcome == AnalysisOutcome.SUCCESS
    result = result.result

    # Verify the result is valid
    assert result is not None
    assert len(result.participant_projections) > 0

    # Verify that only the existing participant ID from keep_participant_ids is actually kept
    # (assuming it meets other clustering criteria)
    clustered_participant_ids = set(
        result.participants_df[result.participants_df["to_cluster"]].index
    )

    # The existing ID should be in the clustered participants if it meets vote threshold
    # The non-existent IDs should be silently ignored (not cause a crash)
    assert max_existing_id + 1000 not in clustered_participant_ids
    assert max_existing_id + 2000 not in clustered_participant_ids


def test_run_pipeline_detects_empty_vote_matrix():
    result = run_pipeline(votes=[])

    assert result.outcome == AnalysisOutcome.INSUFFICIENT_DATA
    assert result.reason == InsufficientDataReason.EMPTY_VOTE_MATRIX


def test_run_pipeline_detects_not_enough_clusterable_participants():
    result = run_pipeline(
        votes=[{"participant_id": 1, "statement_id": 1, "vote": 1}],
        min_user_vote_threshold=1,
    )

    assert result.outcome == AnalysisOutcome.INSUFFICIENT_DATA
    assert result.reason == InsufficientDataReason.NOT_ENOUGH_CLUSTERABLE_PARTICIPANTS


def test_run_pipeline_detects_not_enough_samples_for_group_count():
    result = run_pipeline(
        votes=[
            {"participant_id": 1, "statement_id": 1, "vote": 1},
            {"participant_id": 2, "statement_id": 1, "vote": -1},
        ],
        min_user_vote_threshold=1,
        force_group_count=3,
    )

    assert result.outcome == AnalysisOutcome.INSUFFICIENT_DATA
    assert result.reason == InsufficientDataReason.NOT_ENOUGH_SAMPLES_FOR_GROUP_COUNT


def test_run_pipeline_detects_not_enough_unique_points():
    result = run_pipeline(
        votes=[
            {"participant_id": 1, "statement_id": 1, "vote": 1},
            {"participant_id": 2, "statement_id": 1, "vote": 1},
        ],
        min_user_vote_threshold=1,
        force_group_count=2,
    )

    assert result.outcome == AnalysisOutcome.INSUFFICIENT_DATA
    assert result.reason == InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_success(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])

    result = run_pipeline(
        votes=loader.votes_data,
        force_group_count=2,
    )

    assert result.outcome == AnalysisOutcome.SUCCESS
    assert result.result.clusterer is not None


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_can_return_all_kmeans_candidates(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])

    result = run_pipeline(
        votes=loader.votes_data,
        candidate_group_counts=[2, 3],
    )
    assert result.outcome == AnalysisOutcome.SUCCESS
    candidates_result = result.result

    assert [candidate.group_count for candidate in candidates_result.candidates] == [2, 3]
    assert candidates_result.candidates[0].outcome == AnalysisOutcome.SUCCESS
    assert candidates_result.candidates[0].result.clusterer is not None


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_can_return_all_supported_kmeans_candidates(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])

    result = run_pipeline(
        votes=loader.votes_data,
        candidate_group_counts="all",
        max_group_count=4,
    )
    assert result.outcome == AnalysisOutcome.SUCCESS

    assert [candidate.group_count for candidate in result.result.candidates] == [2, 3, 4]


def test_run_pipeline_rejects_ambiguous_candidate_group_count_list():
    with pytest.raises(ValueError, match="use force_group_count"):
        run_pipeline(
            votes=[
                {"participant_id": 1, "statement_id": 1, "vote": 1},
                {"participant_id": 2, "statement_id": 1, "vote": -1},
            ],
            min_user_vote_threshold=1,
            candidate_group_counts=[2],
        )


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_pca_projection_can_be_reused_for_forced_kmeans(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            f"{fixture.data_dir}/comments.json",
        ]
    )
    _, _, mod_out_statement_ids, meta_statement_ids = process_statements(
        statement_data=loader.comments_data
    )
    force_group_count = 2

    projection = prepare_pca_projection(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
    )
    reused = run_kmeans_on_pca_projection(
        projection=projection,
        force_group_count=force_group_count,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    direct = run_pipeline(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
        force_group_count=force_group_count,
    )
    assert direct.outcome == AnalysisOutcome.SUCCESS
    direct = direct.result
    assert reused.outcome == AnalysisOutcome.SUCCESS
    reused = reused.result

    assert_frame_equal(
        reused.participants_df.loc[:, ["x", "y", "to_cluster", "cluster_id"]],
        direct.participants_df.loc[:, ["x", "y", "to_cluster", "cluster_id"]],
    )
    assert_frame_equal(reused.statements_df, direct.statements_df)


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_projection_silhouette_score_api(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    projection = prepare_pca_projection(votes=loader.votes_data)
    result = run_kmeans_on_pca_projection(
        projection=projection,
        force_group_count=2,
    )
    assert result.outcome == AnalysisOutcome.SUCCESS
    result = result.result

    score = calculate_projection_silhouette_score(
        projection=projection,
        clusterer_model=result.clusterer,
    )

    assert score is not None
