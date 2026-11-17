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
def test_select_representative_statements_excludes_no_directional_votes(vote):
    """Fallback selection must not manufacture agree/disagree evidence."""
    vote_matrix = pd.DataFrame({7: [vote] * 10})
    grouped_stats, _ = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=[0] * 5 + [1] * 5,
        consensus_mode="legacy",
    )

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)

    assert result == {0: [], 1: []}


def test_select_representative_statements_keeps_voted_fallback():
    """An insufficient but voted statement is still eligible for fallback."""
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
        assert statement["tid"] == 8
        assert statement["n-success"] == 1
        assert statement["n-trials"] == 2
        assert "best-agree" not in statement


def test_select_representative_statements_best_agree_rejects_disagreement(
    mixed_direction_data,
):
    """Two negative z-scores must not turn disagreement into best-agree."""
    _, grouped_stats = mixed_direction_data

    result = stats.select_representative_statements(grouped_stats_df=grouped_stats)
    best_agrees = [
        statement for statement in result[0] if statement.get("best-agree", False)
    ]

    assert len(best_agrees) == 1
    assert best_agrees[0]["tid"] == 7
    assert best_agrees[0]["repful-for"] == "agree"
    assert any(
        statement["tid"] == 8 and statement["repful-for"] == "disagree"
        for statement in result[0]
    )


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
    assert statement["n-success"] == expected_agrees


def test_select_representative_statements_shared_agreement_keeps_agree_fields():
    """Best-agree describes agreement even when groups agree equally."""
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
        assert statement["repful-for"] == "agree"
        assert statement["n-agree"] == 20
        assert statement["n-success"] == 20
        assert statement["p-success"] == pytest.approx(21 / 22)


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
