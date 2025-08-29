import pytest
import numpy as np
from reddwarf.implementations.base import run_pipeline
from reddwarf.data_loader import Loader
from tests.fixtures import polis_convo_data


## Determinism via random_seed for run_pipeline()


@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_deterministic_with_random_state(reducer, polis_convo_data):
    """Test that setting random_state results in the same participant_projections twice in a row."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    random_state = 42

    # Run pipeline twice with same random_state
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=random_state)

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=random_state)

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
def test_run_pipeline_not_deterministic_without_random_state(reducer, polis_convo_data):
    """Test that not setting random_state results in different participant_projections across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Run pipeline twice without random_state (should be non-deterministic)
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=None)

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=None)

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
    reducer, polis_convo_data
):
    """Test that not setting random_state for PCA is still deterministic across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = loader.votes_data

    # Run pipeline twice without random_state (PCA should still be deterministic)
    result_1 = run_pipeline(votes=votes, reducer=reducer, random_state=None)

    result_2 = run_pipeline(votes=votes, reducer=reducer, random_state=None)

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
