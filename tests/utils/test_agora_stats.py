import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from reddwarf.utils.stats import (
    agora_label_candidate_mask,
    choose_agora_thresholded_winners,
    benjamini_hochberg,
    calculate_comment_statistics_dataframes,
    classify_signal_strength,
    rank_representative_statements,
    z_to_pvalue,
)
from reddwarf.utils.consensus import rank_consensus_statements
from reddwarf.implementations.agora import compute_effective_agreement_gac


# --- z_to_pvalue ---


def test_z_to_pvalue_zero():
    assert z_to_pvalue(0.0) == pytest.approx(0.5)


def test_z_to_pvalue_large_positive():
    p = z_to_pvalue(5.0)
    assert p < 1e-5


def test_z_to_pvalue_negative():
    p = z_to_pvalue(-2.0)
    assert p > 0.5


def test_z_to_pvalue_array():
    result = z_to_pvalue(np.array([0.0, 5.0, -2.0]))
    assert result[0] == pytest.approx(0.5)
    assert result[1] < 1e-5
    assert result[2] > 0.5


# --- benjamini_hochberg ---


def test_benjamini_hochberg_basic():
    # 5 hypotheses, first 2 have small p-values
    p_values = np.array([0.001, 0.01, 0.3, 0.5, 0.9])
    selected = benjamini_hochberg(p_values, fdr_rate=0.10)
    # First two should be selected
    assert selected[0] == True
    assert selected[1] == True
    # Rest should not
    assert selected[2] == False
    assert selected[3] == False
    assert selected[4] == False


def test_benjamini_hochberg_empty():
    result = benjamini_hochberg(np.array([]), fdr_rate=0.10)
    assert len(result) == 0


def test_benjamini_hochberg_all_significant():
    p_values = np.array([0.001, 0.002, 0.003, 0.004])
    selected = benjamini_hochberg(p_values, fdr_rate=0.10)
    assert all(selected)


def test_benjamini_hochberg_none_significant():
    p_values = np.array([0.5, 0.6, 0.7, 0.8])
    selected = benjamini_hochberg(p_values, fdr_rate=0.10)
    assert not any(selected)


def test_benjamini_hochberg_adapts_to_size():
    """With many hypotheses, BH should be more conservative than a fixed threshold."""
    rng = np.random.default_rng(42)
    # 1000 p-values, 50 truly significant (very small) + 950 null (uniform)
    p_true = rng.uniform(1e-8, 1e-4, size=50)
    p_null = rng.uniform(0.05, 1.0, size=950)
    p_values = np.concatenate([p_true, p_null])

    selected_bh = benjamini_hochberg(p_values, fdr_rate=0.10)
    # Fixed threshold at 0.05 would select everything below 0.05
    selected_fixed = p_values < 0.05

    # BH should select fewer (or equal) than fixed threshold
    assert selected_bh.sum() <= selected_fixed.sum()
    # BH should still find some of the truly significant ones
    assert selected_bh.sum() > 0


# --- rank_representative_statements ---


def _make_three_statement_vote_matrix():
    vote_matrix = pd.DataFrame(
        {
            0: [1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            2: [1, 1, -1, -1, -1, -1, 0, 0, np.nan, np.nan,
                -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        },
        index=list(range(20)),
    )
    cluster_labels = [0] * 10 + [1] * 10
    return vote_matrix, cluster_labels


def _make_two_group_single_statement_vote_matrix(group0_votes, group1_votes):
    vote_matrix = pd.DataFrame(
        {0: list(group0_votes) + list(group1_votes)},
        index=list(range(len(group0_votes) + len(group1_votes))),
    )
    cluster_labels = [0] * len(group0_votes) + [1] * len(group1_votes)
    return vote_matrix, cluster_labels


def _rank_vote_matrix(vote_matrix, cluster_labels, mod_out_statement_ids=None):
    grouped_stats_df, _ = calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )
    return rank_representative_statements(
        grouped_stats_df=grouped_stats_df,
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
        mod_out_statement_ids=mod_out_statement_ids or [],
        fdr_rate=0.10,
        divisive_n_resamples=99,
        divisive_random_state=7,
    )


def test_rank_representative_statements_requires_cluster_context():
    with pytest.raises(ValueError):
        rank_representative_statements(pd.DataFrame())


def test_rank_representative_statements_requires_at_least_two_groups():
    vote_matrix = pd.DataFrame({0: [1, -1, 1]}, index=[0, 1, 2])
    cluster_labels = [0, 0, 0]
    grouped_stats_df, _ = calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )

    with pytest.raises(
        ValueError,
        match="Agora representative ranking requires at least 2 distinct groups",
    ):
        rank_representative_statements(
            grouped_stats_df=grouped_stats_df,
            vote_matrix=vote_matrix,
            cluster_labels=cluster_labels,
        )


def test_rank_representative_statements_all_present():
    vote_matrix, cluster_labels = _make_three_statement_vote_matrix()
    result = _rank_vote_matrix(vote_matrix, cluster_labels)
    assert set(result.keys()) == {0, 1}
    for gid in [0, 1]:
        statement_ids = {s.statement_id for s in result[gid]}
        assert statement_ids == {0, 1, 2}


def test_rank_representative_statements_ranking_order():
    vote_matrix, cluster_labels = _make_three_statement_vote_matrix()
    result = _rank_vote_matrix(vote_matrix, cluster_labels)
    for gid in result:
        statements = result[gid]
        effect_sizes = [s.effect_size for s in statements]
        assert effect_sizes == sorted(effect_sizes, reverse=True)
        assert statements[0].rank == 1


def test_rank_representative_statements_single_eligible_label_wins():
    result = _rank_vote_matrix(*_make_two_group_single_statement_vote_matrix(
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ))
    stmt = [s for s in result[0] if s.statement_id == 0][0]
    assert stmt.repful_for == "agree"


def test_rank_representative_statements_multiple_eligible_labels_use_highest_effect():
    result = _rank_vote_matrix(*_make_two_group_single_statement_vote_matrix(
        [1, 1, -1, -1, -1, -1, 0, 0, np.nan, np.nan],
        [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    ))
    stmt = [s for s in result[0] if s.statement_id == 0][0]
    assert stmt.repful_for == "divisive"


def test_rank_representative_statements_no_eligible_labels_fall_back_to_strongest_local_direction():
    result = _rank_vote_matrix(*_make_two_group_single_statement_vote_matrix(
        [1, 1, 1, -1, 0, 0, 0, 0, np.nan, np.nan],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ))
    stmt = [s for s in result[0] if s.statement_id == 0][0]
    assert stmt.repful_for == "agree"


def test_rank_representative_statements_one_by_one_divisive_is_blocked():
    result = _rank_vote_matrix(*_make_two_group_single_statement_vote_matrix(
        [1, -1, 0, 0, np.nan, np.nan],
        [1, 1, 1, 1, 1, 1],
    ))
    stmt = [s for s in result[0] if s.statement_id == 0][0]
    assert stmt.repful_for != "divisive"


def test_rank_representative_statements_ties_break_by_lower_p_value_then_label_order():
    winners = choose_agora_thresholded_winners(
        label_effects=[[1.0, 1.0, 0.5], [0.4, 0.4, 0.1], [0.0, 0.0, 0.0]],
        label_p_values=[[0.2, 0.1, 0.9], [0.2, 0.1, 1.0], [0.1, 0.1, 1.0]],
        candidate_mask=[[True, True, False], [False, False, False], [False, False, False]],
        fallback_local_scores=[[0.0, 0.0, 0.0], [0.3, 0.3, -np.inf], [0.3, 0.3, -np.inf]],
    )
    assert_array_equal(winners, np.array([1, 1, 0]))


def test_rank_representative_statements_candidate_mask_blocks_ultrathin_divisive():
    mask = agora_label_candidate_mask(
        n_agree=[1],
        n_disagree=[1],
        n_seen=[2],
    )[0]
    assert_array_equal(mask, np.array([True, True, False]))


def test_rank_representative_statements_zero_vote_filter():
    """Zero-vote statements should not inflate BH hypothesis count."""
    vote_matrix, cluster_labels = _make_three_statement_vote_matrix()
    vote_matrix_with_zero = vote_matrix.copy()
    vote_matrix_with_zero[99] = np.nan

    result_with = _rank_vote_matrix(vote_matrix_with_zero, cluster_labels)
    result_without = _rank_vote_matrix(vote_matrix, cluster_labels)

    for gid in result_with:
        zero_stmt = [s for s in result_with[gid] if s.statement_id == 99][0]
        assert zero_stmt.selected is False
        assert zero_stmt.adjusted_p_value == 1.0
        sel_with = {s.statement_id for s in result_with[gid] if s.selected and s.statement_id != 99}
        sel_without = {s.statement_id for s in result_without[gid] if s.selected}
        assert sel_with == sel_without


def test_classify_signal_strength_small_group_requires_full_participation():
    assert classify_signal_strength(
        selected=True,
        effect_size=2.0,
        p_value=0.01,
        n_seen=4,
        group_size=4,
        strong_effect_min=1.0,
        strong_small_group_cutoff=5,
        strong_large_group_participation_min=0.8,
        strong_p_max=0.05,
    ) == "strong"


def test_classify_signal_strength_small_group_partial_participation_is_normal():
    assert classify_signal_strength(
        selected=True,
        effect_size=2.0,
        p_value=0.01,
        n_seen=3,
        group_size=4,
        strong_effect_min=1.0,
        strong_small_group_cutoff=5,
        strong_large_group_participation_min=0.8,
        strong_p_max=0.05,
    ) == "normal"


def test_classify_signal_strength_group_of_five_uses_eighty_percent_rule():
    assert classify_signal_strength(
        selected=True,
        effect_size=2.0,
        p_value=0.01,
        n_seen=4,
        group_size=5,
        strong_effect_min=1.0,
        strong_small_group_cutoff=5,
        strong_large_group_participation_min=0.8,
        strong_p_max=0.05,
    ) == "strong"


def test_classify_signal_strength_large_group_low_participation_is_normal():
    assert classify_signal_strength(
        selected=True,
        effect_size=2.0,
        p_value=0.01,
        n_seen=5,
        group_size=50,
        strong_effect_min=1.0,
        strong_small_group_cutoff=5,
        strong_large_group_participation_min=0.8,
        strong_p_max=0.05,
    ) == "normal"


def test_classify_signal_strength_non_positive_group_size_is_normal():
    assert classify_signal_strength(
        selected=True,
        effect_size=2.0,
        p_value=0.01,
        n_seen=4,
        group_size=0,
        strong_effect_min=1.0,
        strong_small_group_cutoff=5,
        strong_large_group_participation_min=0.8,
        strong_p_max=0.05,
    ) == "normal"


# --- rank_consensus_statements ---


def _make_consensus_vote_matrix():
    """Create a synthetic vote matrix for consensus testing."""
    # 10 voters, 4 statements
    return pd.DataFrame(
        {
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],      # strong agree
            1: [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],  # strong disagree
            2: [1, 1, 1, -1, -1, -1, 0, 0, 0, 0],    # mixed
            3: [1, 1, 1, 1, 1, 1, -1, -1, -1, -1],   # moderate agree
        },
        index=list(range(10)),
    )


def test_rank_consensus_statements_all_present():
    vm = _make_consensus_vote_matrix()
    result = rank_consensus_statements(vm)
    agree_ids = {s.statement_id for s in result.agree}
    disagree_ids = {s.statement_id for s in result.disagree}
    # All 4 statements should appear in both directions
    assert agree_ids == {0, 1, 2, 3}
    assert disagree_ids == {0, 1, 2, 3}


def test_rank_consensus_statements_ranking():
    vm = _make_consensus_vote_matrix()
    result = rank_consensus_statements(vm)
    # Agree: ranked by pa descending
    agree_pa = [s.pa for s in result.agree]
    assert agree_pa == sorted(agree_pa, reverse=True)
    # Disagree: ranked by pd descending
    disagree_pd = [s.pd for s in result.disagree]
    assert disagree_pd == sorted(disagree_pd, reverse=True)


def test_rank_consensus_statements_effect_size_matches_probability():
    vm = _make_consensus_vote_matrix()
    result = rank_consensus_statements(vm)
    for s in result.agree:
        assert s.effect_size == pytest.approx(s.pa)
    for s in result.disagree:
        assert s.effect_size == pytest.approx(s.pd)


# --- effective agreement GAC ---


def test_effective_agreement_penalizes_divided_groups():
    """A group split between agree/disagree should lower GAC vs unanimous."""
    # 2 groups of 5 participants, 1 statement.
    # Group 0: all agree. Group 1: split 3 agree / 2 disagree.
    vote_matrix_divided = pd.DataFrame(
        {0: [1, 1, 1, 1, 1, 1, 1, 1, -1, -1]},
        index=list(range(10)),
    )
    cluster_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    vote_matrix_unanimous = pd.DataFrame(
        {0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        index=list(range(10)),
    )

    grouped_divided, _ = calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix_divided, cluster_labels=cluster_labels,
    )
    grouped_unanimous, _ = calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix_unanimous, cluster_labels=cluster_labels,
    )

    gac_divided = compute_effective_agreement_gac(grouped_divided, [0], n_groups=2)
    gac_unanimous = compute_effective_agreement_gac(grouped_unanimous, [0], n_groups=2)

    # Divided group should lower the score
    assert gac_divided["agree"][0] < gac_unanimous["agree"][0]
    # A group split 3/2 should produce a score below 0.5
    assert gac_divided["agree"][0] < 0.5
    # Unanimous agreement should be well above 0.5
    assert gac_unanimous["agree"][0] > 0.5


def test_effective_agreement_gac_bounded():
    """All GAC scores should be in [0, 1] and <= raw pa baseline."""
    vote_matrix = pd.DataFrame(
        {
            0: [1, 1, 1, -1, -1, 1],
            1: [-1, -1, 1, 1, 1, -1],
            2: [1, 1, 1, 1, 1, 1],
        },
        index=list(range(6)),
    )
    cluster_labels = [0, 0, 0, 1, 1, 1]

    grouped_stats_df, _ = calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix, cluster_labels=cluster_labels,
    )
    n_groups = 2
    statement_ids = vote_matrix.columns.tolist()

    gac = compute_effective_agreement_gac(grouped_stats_df, statement_ids, n_groups)

    for sid in statement_ids:
        ea = gac["agree"][sid]
        ed = gac["disagree"][sid]
        assert 0 <= ea <= 1, f"agree GAC {ea} out of [0,1] for statement {sid}"
        assert 0 <= ed <= 1, f"disagree GAC {ed} out of [0,1] for statement {sid}"

        # Effective agreement should be <= raw pa geometric mean
        pa_product = 1.0
        for gid in range(n_groups):
            pa = grouped_stats_df.loc[(gid, sid), "pa"]
            pa_product *= pa
        raw_baseline = pa_product ** (1.0 / n_groups)
        assert ea <= raw_baseline + 1e-9, (
            f"Statement {sid}: effective agreement {ea} > raw baseline {raw_baseline}"
        )
