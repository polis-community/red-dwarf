from types import SimpleNamespace

import pandas as pd

from reddwarf.implementations import agora


def _mock_base_result() -> SimpleNamespace:
    raw_vote_matrix = pd.DataFrame({0: [1, -1]}, index=[10, 11])
    participants_df = pd.DataFrame(
        {
            "to_cluster": [True, True],
            "cluster_id": [0, 1],
        },
        index=raw_vote_matrix.index,
    )
    group_comment_stats = pd.DataFrame(
        {
            "pa": [0.8, 0.2],
            "pd": [0.2, 0.8],
        },
        index=pd.MultiIndex.from_tuples(
            [(0, 0), (1, 0)], names=["group_id", "statement_id"]
        ),
    )
    return SimpleNamespace(
        raw_vote_matrix=raw_vote_matrix,
        filtered_vote_matrix=raw_vote_matrix,
        reducer=None,
        clusterer=None,
        group_comment_stats=group_comment_stats,
        statements_df=pd.DataFrame(index=[0]),
        participants_df=participants_df,
        participant_projections={},
        statement_projections=None,
    )


def test_run_pipeline_keeps_divisive_random_state_independent(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(agora.base, "run_pipeline", lambda **kwargs: _mock_base_result())
    monkeypatch.setattr(
        agora,
        "rank_representative_statements",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        agora,
        "rank_consensus_statements",
        lambda **kwargs: {"agree": [], "disagree": []},
    )
    monkeypatch.setattr(
        agora,
        "compute_effective_agreement_gac",
        lambda *args, **kwargs: {"agree": {}, "disagree": {}},
    )

    agora.run_pipeline(votes=[], random_state=42)

    assert captured["divisive_random_state"] is None


def test_run_pipeline_passes_explicit_divisive_random_state(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(agora.base, "run_pipeline", lambda **kwargs: _mock_base_result())
    monkeypatch.setattr(
        agora,
        "rank_representative_statements",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    monkeypatch.setattr(
        agora,
        "rank_consensus_statements",
        lambda **kwargs: {"agree": [], "disagree": []},
    )
    monkeypatch.setattr(
        agora,
        "compute_effective_agreement_gac",
        lambda *args, **kwargs: {"agree": {}, "disagree": {}},
    )

    agora.run_pipeline(votes=[], random_state=42, divisive_random_state=7)

    assert captured["divisive_random_state"] == 7
