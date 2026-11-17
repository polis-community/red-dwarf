from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from reddwarf.implementations import agora, polis
from reddwarf.implementations.agora import AgoraClusteringResult
from reddwarf.implementations.base import (
    AnalysisOutcome,
    PolisClusteringResult,
    prepare_pca_projection,
)


@dataclass(frozen=True)
class PipelineSelectionResults:
    polis_result: PolisClusteringResult
    agora_result: AgoraClusteringResult
    polis_group_id: int
    agora_group_id: int


def _run_pipeline_pair(vote_matrix: pd.DataFrame) -> PipelineSelectionResults:
    votes = [
        {
            "participant_id": int(participant_id),
            "statement_id": int(statement_id),
            "vote": int(vote),
        }
        for (participant_id, statement_id), vote in vote_matrix.stack().dropna().items()
    ]
    projection = prepare_pca_projection(votes=votes, random_state=42)
    coordinates = projection.participants_df.loc[:, ["x", "y"]]

    # Seed the fixture's two populations to isolate selection from initialization.
    # Both public pipelines still run their actual PCA and k-means implementations.
    init_centers = [
        coordinates.iloc[:20].mean().tolist(),
        coordinates.iloc[20:].mean().tolist(),
    ]
    polis_outcome = polis.run_pipeline(
        votes=votes,
        force_group_count=2,
        init_centers=init_centers,
        random_state=42,
    )
    agora_outcome = agora.run_pipeline(
        votes=votes,
        force_group_count=2,
        init_centers=init_centers,
        random_state=42,
    )
    assert polis_outcome.outcome == AnalysisOutcome.SUCCESS
    assert agora_outcome.outcome == AnalysisOutcome.SUCCESS
    assert isinstance(polis_outcome.result, PolisClusteringResult)
    assert isinstance(agora_outcome.result, AgoraClusteringResult)
    polis_result = polis_outcome.result
    agora_result = agora_outcome.result
    polis_group_id = int(polis_result.participants_df.loc[0, "cluster_id"])
    agora_group_id = int(agora_result.participants_df.loc[0, "cluster_id"])

    for result in (polis_result, agora_result):
        participants = result.participants_df
        group_id = int(participants.loc[0, "cluster_id"])
        assert participants["to_cluster"].all()
        assert set(participants.index) == set(range(40))
        assert set(
            participants.index[participants["cluster_id"] == group_id]
        ) == set(range(20))
        assert participants["cluster_id"].nunique() == 2

    return PipelineSelectionResults(
        polis_result=polis_result,
        agora_result=agora_result,
        polis_group_id=polis_group_id,
        agora_group_id=agora_group_id,
    )


def test_agora_pipeline_selects_strong_disagreement_without_agreement_headline():
    vote_matrix = pd.DataFrame(
        {statement_id: [-1] * 20 + [1] * 20 for statement_id in range(8)}
    )

    pair = _run_pipeline_pair(vote_matrix)
    legacy_statements = pair.polis_result.repness[pair.polis_group_id]
    ranked_statements = pair.agora_result.ranked_repness[pair.agora_group_id]
    target = next(
        statement for statement in ranked_statements if statement.statement_id == 7
    )

    assert legacy_statements
    assert all(statement["repful-for"] == "disagree" for statement in legacy_statements)
    assert all("best-agree" not in statement for statement in legacy_statements)
    assert target.repful_for == "disagree"
    assert target.selected is True
    assert target.na == 0
    assert target.nd == 20
    assert target.ns == 20


def test_agora_pipeline_reports_disagreement_without_legacy_best_agree_metadata():
    vote_matrix = pd.DataFrame(
        {
            **{statement_id: [-1] * 20 + [1] * 20 for statement_id in range(7)},
            7: [1] * 10 + [-1] * 9 + [np.nan] + [1] * 10 + [-1] * 10,
            8: [-1] * 20 + [1] * 20,
        }
    )

    pair = _run_pipeline_pair(vote_matrix)
    legacy_target = next(
        statement
        for statement in pair.polis_result.repness[pair.polis_group_id]
        if statement["tid"] == 8
    )
    agora_target = next(
        statement
        for statement in pair.agora_result.ranked_repness[pair.agora_group_id]
        if statement.statement_id == 8
    )

    assert legacy_target["best-agree"] is True
    assert legacy_target["repful-for"] == "disagree"
    assert legacy_target["n-agree"] == 0
    assert legacy_target["n-success"] == 20
    assert agora_target.repful_for == "disagree"
    assert agora_target.selected is True
    assert agora_target.na == 0
    assert agora_target.nd == 20
    assert agora_target.ns == 20


@pytest.mark.parametrize("vote", [0, np.nan], ids=["pass-only", "unvoted-in-group"])
def test_agora_pipeline_does_not_select_without_directional_votes(vote):
    vote_matrix = pd.DataFrame(
        {
            **{statement_id: [0] * 20 + [1] * 20 for statement_id in range(7)},
            7: [vote] * 20 + [1] * 20,
        }
    )

    pair = _run_pipeline_pair(vote_matrix)
    legacy_statements = pair.polis_result.repness[pair.polis_group_id]
    ranked_statements = pair.agora_result.ranked_repness[pair.agora_group_id]

    assert len(legacy_statements) == 1
    assert legacy_statements[0]["repful-for"] == "disagree"
    assert legacy_statements[0]["n-success"] == 0
    assert {statement.statement_id for statement in ranked_statements} == set(range(8))
    for statement in ranked_statements:
        assert statement.na == 0
        assert statement.nd == 0
        assert statement.selected is False
        assert statement.adjusted_p_value == 1.0

    target = next(
        statement for statement in ranked_statements if statement.statement_id == 7
    )
    assert target.ns == (20 if vote == 0 else 0)


def test_agora_pipeline_labels_shared_agreement_as_agree():
    """Check direction and counts, not evidence of cross-group distinctiveness."""
    vote_matrix = pd.DataFrame(
        {
            **{statement_id: [-1] * 20 + [1] * 20 for statement_id in range(7)},
            7: [1] * 40,
        }
    )

    pair = _run_pipeline_pair(vote_matrix)
    legacy_target = next(
        statement
        for statement in pair.polis_result.repness[pair.polis_group_id]
        if statement["tid"] == 7
    )
    agora_target = next(
        statement
        for statement in pair.agora_result.ranked_repness[pair.agora_group_id]
        if statement.statement_id == 7
    )

    assert legacy_target["best-agree"] is True
    assert legacy_target["repful-for"] == "disagree"
    assert legacy_target["n-agree"] == 20
    assert legacy_target["n-success"] == 0
    assert agora_target.repful_for == "agree"
    assert agora_target.na == 20
    assert agora_target.nd == 0
    assert agora_target.ns == 20
