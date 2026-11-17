"""Polis port regressions and intentional legacy-behavior characterization.

Expected legacy behavior follows beats-best-agr?, finalize-cmt-stats, and
select-rep-comments in upstream math/src/polismath/math/repness.clj:
https://github.com/compdemocracy/polis/blob/7bf9eccc287586e51d96fdf519ae6da98e0f4a70/math/src/polismath/math/repness.clj
"""

import numpy as np
import pandas as pd
import pytest

from reddwarf.utils import stats


@pytest.fixture
def mixed_direction_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    vote_matrix = pd.DataFrame(
        {
            7: [1] * 10 + [-1] * 9 + [np.nan] + [1] * 10 + [-1] * 10,
            8: [-1] * 20 + [1] * 20,
        }
    )
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 20 + [1] * 20,
        consensus_mode="legacy",
    )
    return vote_matrix, grouped_stats


def test_select_representative_statements_keeps_disagreement_without_best_agree():
    """A significant disagreement must not require an agreement headline."""
    vote_matrix = pd.DataFrame({7: [-1] * 20 + [1] * 20})
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 20 + [1] * 20,
        consensus_mode="legacy",
    )

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)

    assert [statement["tid"] for statement in result[0]] == [7]
    assert result[0][0]["repful-for"] == "disagree"
    assert result[0][0]["n-success"] == 20
    assert "best-agree" not in result[0][0]
    assert result[1][0]["repful-for"] == "agree"


@pytest.mark.parametrize("vote", [0, np.nan], ids=["pass-only", "unvoted"])
def test_legacy_fallback_can_select_without_directional_votes(vote):
    """Characterize upstream fallback, not desirable Agora selection semantics."""
    vote_matrix = pd.DataFrame({7: [vote] * 10})
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 5 + [1] * 5,
        consensus_mode="legacy",
    )

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)

    expected_trials = 5 if vote == 0 else 0
    for statements in result.values():
        assert len(statements) == 1
        statement = statements[0]
        assert statement["tid"] == 7
        assert statement["repful-for"] == "disagree"
        assert statement["n-success"] == 0
        assert statement["n-trials"] == expected_trials
        assert statement["p-success"] == pytest.approx(1 / (expected_trials + 2))
        assert "best-agree" not in statement


def test_legacy_fallback_keeps_first_candidate_when_repness_tests_tie():
    """The legacy fallback can retain a pass-only candidate before a voted one."""
    vote_matrix = pd.DataFrame(
        {
            7: [0] * 8,
            8: [1, -1, np.nan, np.nan] * 2,
        }
    )
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 4 + [1] * 4,
        consensus_mode="legacy",
    )

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)

    for statements in result.values():
        assert len(statements) == 1
        statement = statements[0]
        assert statement["tid"] == 7
        assert statement["n-success"] == 0
        assert statement["n-trials"] == 4
        assert "best-agree" not in statement


def test_legacy_best_agree_can_select_unanimous_disagreement(
    mixed_direction_data,
):
    """Preserve upstream's multiplication of two negative agreement z-scores."""
    _, grouped_stats = mixed_direction_data

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)
    best_agrees = [
        statement for statement in result[0] if statement.get("best-agree", False)
    ]

    assert len(best_agrees) == 1
    assert best_agrees[0]["tid"] == 8
    assert best_agrees[0]["repful-for"] == "disagree"
    assert best_agrees[0]["n-success"] == 20


def test_select_representative_statements_best_agree_count_matches_votes(
    mixed_direction_data,
):
    vote_matrix, grouped_stats = mixed_direction_data

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)
    best_agrees = [
        statement for statement in result[0] if statement.get("best-agree", False)
    ]

    assert len(best_agrees) == 1
    statement = best_agrees[0]
    expected_agrees = int((vote_matrix.loc[:19, statement["tid"]] == 1).sum())
    assert statement["n-agree"] == expected_agrees
    assert statement["n-success"] == 20
    assert statement["n-agree"] != statement["n-success"]


def test_legacy_best_agree_preserves_disagree_direction_on_equal_repness():
    """Upstream finalize-cmt-stats chooses disagree when rat and rdt tie."""
    vote_matrix = pd.DataFrame({7: [1] * 40})
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 20 + [1] * 20,
        consensus_mode="legacy",
    )

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)

    for statements in result.values():
        assert len(statements) == 1
        statement = statements[0]
        assert statement["best-agree"] is True
        assert statement["repful-for"] == "disagree"
        assert statement["n-agree"] == 20
        assert statement["n-success"] == 0
        assert statement["p-success"] == pytest.approx(1 / 22)


def test_select_representative_statements_respects_moderation(mixed_direction_data):
    _, grouped_stats = mixed_direction_data

    result = stats.select_representative_statements(
        grouped_stats_df=grouped_stats,
        mod_out_statement_ids=[7],
    )

    assert [statement["tid"] for statement in result[0]] == [8]
    assert result[0][0]["repful-for"] == "disagree"
    assert [statement["tid"] for statement in result[1]] == [8]


@pytest.mark.parametrize("pick_max", [1, 2])
def test_select_representative_statements_caps_and_deduplicates(
    mixed_direction_data, pick_max
):
    _, grouped_stats = mixed_direction_data

    result = stats.select_representative_statements(
        grouped_stats_df=grouped_stats,
        pick_max=pick_max,
    )

    for statements in result.values():
        statement_ids = [statement["tid"] for statement in statements]
        assert len(statement_ids) <= pick_max
        assert len(statement_ids) == len(set(statement_ids))
        assert statements[0]["best-agree"] is True
