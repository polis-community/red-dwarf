import pytest

from reddwarf.data_loader import Loader
from reddwarf.implementations.agora import run_pipeline
from reddwarf.implementations.base import AnalysisOutcome, InsufficientDataReason
from tests.fixtures import polis_convo_data


def test_agora_run_pipeline_returns_base_insufficient_data_reason():
    result = run_pipeline(votes=[])

    assert result.outcome == AnalysisOutcome.INSUFFICIENT_DATA
    assert result.reason == InsufficientDataReason.EMPTY_VOTE_MATRIX


@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_agora_run_pipeline_can_return_all_kmeans_candidates(polis_convo_data):
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
    assert candidates_result.candidates[0].result.ranked_repness
