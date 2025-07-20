import pytest
import numpy as np
from reddwarf.implementations.base import run_pipeline
from reddwarf.data_loader import Loader
from tests.fixtures import polis_convo_data
from reddwarf.utils.string_id import preprocess_votes


## Determinism via random_seed for run_pipeline()

@pytest.mark.parametrize("reducer", ["pca", "pacmap", "localmap"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_deterministic_with_random_state(reducer, polis_convo_data):
    """Test that setting random_state results in the same participant_projections twice in a row."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes =  preprocess_votes(loader.votes_data)


    random_state = 42

    # Run pipeline twice with same random_state
    result_1 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=random_state
    )

    result_2 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=random_state
    )

    # Results should be identical when using same random_state
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array([result_1.participant_projections[pid] for pid in participant_ids])
    projections_2 = np.array([result_2.participant_projections[pid] for pid in participant_ids])

    np.testing.assert_array_equal(
        projections_1,
        projections_2,
        err_msg=f"{reducer} with random_state should produce identical participant projections"
    )


@pytest.mark.parametrize("reducer", ["pacmap", "localmap"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_not_deterministic_without_random_state(reducer, polis_convo_data):
    """Test that not setting random_state results in different participant_projections across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = preprocess_votes(loader.votes_data)

    # Run pipeline twice without random_state (should be non-deterministic)
    result_1 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=None
    )

    result_2 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=None
    )

    # Results should be different when not using random_state
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array([result_1.participant_projections[pid] for pid in participant_ids])
    projections_2 = np.array([result_2.participant_projections[pid] for pid in participant_ids])

    # We use assert_raises to expect that arrays are NOT equal
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            projections_1,
            projections_2,
            err_msg=f"{reducer} without random_state should produce different participant projections"
        )


# Note: PCA is always deterministic regardless of random_state
@pytest.mark.parametrize("reducer", ["pca"])
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_pipeline_pca_still_deterministic_without_random_state(reducer, polis_convo_data):
    """Test that not setting random_state for PCA is still deterministic across runs."""
    fixture = polis_convo_data

    # Load test data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    votes = preprocess_votes(loader.votes_data)

    # Run pipeline twice without random_state (PCA should still be deterministic)
    result_1 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=None
    )

    result_2 = run_pipeline(
        votes=votes,
        reducer=reducer,
        random_state=None
    )

    # Results should be identical even without random_state for PCA
    # Convert participant_projections dict to arrays for comparison
    participant_ids = sorted(result_1.participant_projections.keys())
    projections_1 = np.array([result_1.participant_projections[pid] for pid in participant_ids])
    projections_2 = np.array([result_2.participant_projections[pid] for pid in participant_ids])

    np.testing.assert_array_equal(
        projections_1,
        projections_2,
        err_msg=f"{reducer} should produce identical participant projections even without random_state"
    )