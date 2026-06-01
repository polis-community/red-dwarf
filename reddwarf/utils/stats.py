import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Optional, Literal
from types import SimpleNamespace
from scipy.stats import norm, permutation_test
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.types.polis import (
    PolisRepness,
    PolisRepnessStatement,
)
from reddwarf.types.agora import RankedRepnessStatement
from reddwarf.utils.reducer.pca import calculate_extremity


def one_prop_test(
    succ: ArrayLike,
    n: ArrayLike,
) -> np.float64 | NDArray[np.float64]:
    # Convert inputs to numpy arrays (if they aren't already)
    succ = np.asarray(succ) + 1
    n = np.asarray(n) + 1

    # Compute the test statistic
    return 2 * np.sqrt(n) * ((succ / n) - 0.5)


# Calculate representativeness two-prop test
def two_prop_test(
    succ_in: ArrayLike,
    succ_out: ArrayLike,
    n_in: ArrayLike,
    n_out: ArrayLike,
) -> np.float64 | NDArray[np.float64]:
    """Two-prop test ported from Polis. Accepts numpy arrays for bulk processing."""
    # Ported with adaptation from Polis
    # See: https://github.com/compdemocracy/polis/blob/90bcb43e67dad660629e0888fedc0d32379f375d/math/src/polismath/math/stats.clj#L18-L33

    succ_in, succ_out, n_in, n_out = map(np.asarray, (succ_in, succ_out, n_in, n_out))
    # Laplace smoothing (add 1 to each count)
    succ_in += 1
    succ_out += 1
    # BUG: Why can't these be switched to += operator and still match polismath output?
    n_in = n_in + 1
    n_out = n_out + 1

    # Compute proportions
    pi1 = succ_in / n_in
    pi2 = succ_out / n_out
    pi_hat = (succ_in + succ_out) / (n_in + n_out)

    denominator = np.sqrt(pi_hat * (1 - pi_hat) * (1 / n_in + 1 / n_out))

    # Compute the test statistic.
    # Suppress divide-by-zero errors, because we correct them below.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (pi1 - pi2) / denominator

    return np.where(
        # Handle edge-case when pi_hat == 1 (would be divide-by-zero error)
        pi_hat == 1,
        # If calculate derivative, the limit is approaching zero here.
        0,
        # Handle everything else.
        result,
    )


def is_significant(z_val: float, confidence: float = 0.90) -> bool:
    """Test whether z-statistic is significant at 90% confidence (one-tailed, right-side)."""
    critical_value = norm.ppf(confidence)  # 90% confidence level, one-tailed
    return z_val > critical_value  # rat/rdt can be negative


def z_to_pvalue(z: ArrayLike) -> np.floating | NDArray[np.floating]:
    """Convert one-tailed right-side z-scores to p-values."""
    return 1 - norm.cdf(np.asarray(z))


def benjamini_hochberg(p_values: NDArray, fdr_rate: float) -> NDArray[np.bool_]:
    """Benjamini-Hochberg procedure for controlling false discovery rate.

    Returns a boolean mask indicating which hypotheses are rejected (selected).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    thresholds = (np.arange(1, n + 1) / n) * fdr_rate
    passing = sorted_p <= thresholds
    if not passing.any():
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(passing))
    selected = np.zeros(n, dtype=bool)
    selected[sorted_indices[: max_k + 1]] = True
    return selected


def apply_bh_with_vote_filter(
    p_values: NDArray,
    has_votes: NDArray[np.bool_],
    fdr_rate: float,
) -> tuple[NDArray[np.bool_], NDArray]:
    """Apply BH selection only to testable statements (those with votes).

    Statements without votes get selected=False and adjusted_p_value=1.0.
    This avoids inflating the hypothesis count with noise from Laplace smoothing.

    Args:
        p_values: Raw p-values for all statements.
        has_votes: Boolean mask — True if the statement has at least one vote.
        fdr_rate: False discovery rate for BH procedure.

    Returns:
        Tuple of (selected_mask, adjusted_p_values) arrays for all statements.
    """
    n = len(p_values)
    selected = np.zeros(n, dtype=bool)
    adjusted = np.ones(n)

    testable = np.where(has_votes)[0]
    if len(testable) == 0:
        return selected, adjusted

    p_testable = p_values[testable]
    selected[testable] = benjamini_hochberg(p_testable, fdr_rate)

    # BH-adjusted p-values for testable statements.
    m = len(p_testable)
    sorted_idx = np.argsort(p_testable)
    sorted_p = p_testable[sorted_idx]
    raw_adj = sorted_p * m / np.arange(1, m + 1)
    cummin = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_testable = np.minimum(cummin, 1.0)
    reordered = np.empty(m)
    reordered[sorted_idx] = adj_testable
    adjusted[testable] = reordered

    return selected, adjusted


def simes_combine(p_value_a: ArrayLike, p_value_b: ArrayLike) -> np.ndarray:
    """Combine two positively dependent directional p-values using Simes' rule."""
    p_value_a = np.asarray(p_value_a, dtype=float)
    p_value_b = np.asarray(p_value_b, dtype=float)
    return np.minimum(
        2 * np.minimum(p_value_a, p_value_b),
        np.maximum(p_value_a, p_value_b),
    )


def divisive_prevalence(
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    n_seen: ArrayLike,
) -> np.ndarray:
    """Return seen-voter divisiveness: 2 * min(agree, disagree) / seen."""
    n_agree, n_disagree, n_seen = map(np.asarray, (n_agree, n_disagree, n_seen))
    balanced_votes = 2 * np.minimum(n_agree, n_disagree)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n_seen > 0, balanced_votes / n_seen, 0.0)


def full_group_prevalence(
    count: ArrayLike,
    group_size: ArrayLike,
) -> np.ndarray:
    """Return whole-group prevalence: count / group_size."""
    count, group_size = map(np.asarray, (count, group_size))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(group_size > 0, count / group_size, 0.0)


def full_group_divisive_prevalence(
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    group_size: ArrayLike,
) -> np.ndarray:
    """Return whole-group divisiveness: 2 * min(agree, disagree) / group_size."""
    n_agree, n_disagree, group_size = map(np.asarray, (n_agree, n_disagree, group_size))
    balanced_votes = 2 * np.minimum(n_agree, n_disagree)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(group_size > 0, balanced_votes / group_size, 0.0)


def smallest_meaningful_prevalence(
    group_size: ArrayLike,
    min_count: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the smallest non-zero prevalence worth treating as real."""
    group_size = np.asarray(group_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(group_size > 0, min_count / group_size, eps)
    return np.clip(floor, eps, 1.0)


def agora_thresholded_effect(
    prevalence_in: ArrayLike,
    prevalence_out: ArrayLike,
    out_group_size: ArrayLike,
    min_out_count: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Score one label by combining whole-group prevalence and comparative distinctiveness."""
    prevalence_in, prevalence_out, out_group_size = map(
        np.asarray, (prevalence_in, prevalence_out, out_group_size)
    )
    prevalence_floor = smallest_meaningful_prevalence(
        out_group_size, min_count=min_out_count, eps=eps
    )
    repness = prevalence_in / np.maximum(prevalence_out, prevalence_floor)
    return prevalence_in * repness


def agora_thresholded_divisive_effect(
    n_agree_in: ArrayLike,
    n_disagree_in: ArrayLike,
    group_size_in: ArrayLike,
    n_agree_out: ArrayLike,
    n_disagree_out: ArrayLike,
    group_size_out: ArrayLike,
    eps: float = 1e-12,
) -> np.ndarray:
    """Score divisive labels using whole-group divisive prevalence in and out of group."""
    divisive_in = full_group_divisive_prevalence(
        n_agree_in,
        n_disagree_in,
        group_size_in,
    )
    divisive_out = full_group_divisive_prevalence(
        n_agree_out,
        n_disagree_out,
        group_size_out,
    )
    return agora_thresholded_effect(
        prevalence_in=divisive_in,
        prevalence_out=divisive_out,
        out_group_size=group_size_out,
        min_out_count=2.0,
        eps=eps,
    )


def agora_label_candidate_mask(
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    n_seen: ArrayLike,
    directional_threshold: float = 0.5,
    divisive_threshold: float = 0.5,
    divisive_min_votes: int = 2,
) -> np.ndarray:
    """Return which labels are locally plausible from the in-group vote pattern alone."""
    n_agree, n_disagree, n_seen = map(np.asarray, (n_agree, n_disagree, n_seen))
    local_scores = agora_local_label_scores(n_agree, n_disagree, n_seen)
    agree_local = local_scores[:, 0]
    disagree_local = local_scores[:, 1]
    divisive_local = local_scores[:, 2]
    divisive_enough_votes = np.minimum(n_agree, n_disagree) >= divisive_min_votes
    return np.column_stack(
        [
            agree_local >= directional_threshold,
            disagree_local >= directional_threshold,
            (divisive_local >= divisive_threshold) & divisive_enough_votes,
        ]
    )


def agora_local_label_scores(
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    n_seen: ArrayLike,
) -> np.ndarray:
    """Return the local in-group strengths for agree, disagree, and divisive."""
    n_agree, n_disagree, n_seen = map(np.asarray, (n_agree, n_disagree, n_seen))
    return np.column_stack(
        [
            full_group_prevalence(n_agree, n_seen),
            full_group_prevalence(n_disagree, n_seen),
            divisive_prevalence(n_agree, n_disagree, n_seen),
        ]
    )


def choose_agora_thresholded_winners(
    label_effects: ArrayLike,
    label_p_values: ArrayLike,
    candidate_mask: ArrayLike,
    fallback_local_scores: ArrayLike,
) -> np.ndarray:
    """Choose final labels from eligible candidates, with local fallback and stable ties."""
    label_effects = np.asarray(label_effects, dtype=float)
    label_p_values = np.asarray(label_p_values, dtype=float)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    fallback_local_scores = np.asarray(fallback_local_scores, dtype=float)

    winners = np.zeros(label_effects.shape[0], dtype=int)
    for idx in range(label_effects.shape[0]):
        allowed = candidate_mask[idx]
        if not allowed.any():
            fallback_indices = np.arange(label_effects.shape[1])
            fallback_scores = fallback_local_scores[idx, fallback_indices]
            best_fallback = fallback_scores.max()
            fallback_winners = fallback_indices[
                np.isclose(fallback_scores, best_fallback)
            ]
            if len(fallback_winners) == 1:
                winners[idx] = fallback_winners[0]
                continue

            winner_p_values = label_p_values[idx, fallback_winners]
            best_p_value = winner_p_values.min()
            p_winners = fallback_winners[np.isclose(winner_p_values, best_p_value)]
            winners[idx] = int(p_winners.min())
            continue

        allowed_indices = np.flatnonzero(allowed)
        allowed_effects = label_effects[idx, allowed_indices]
        best_effect = allowed_effects.max()
        effect_winners = allowed_indices[np.isclose(allowed_effects, best_effect)]

        if len(effect_winners) == 1:
            winners[idx] = effect_winners[0]
            continue

        winner_p_values = label_p_values[idx, effect_winners]
        best_p_value = winner_p_values.min()
        p_winners = effect_winners[np.isclose(winner_p_values, best_p_value)]
        winners[idx] = int(p_winners.min())

    return winners


def _full_group_divisive_statistic(values: ArrayLike) -> float:
    """Scalar whole-group divisiveness statistic used in divisive permutation tests."""
    values = np.asarray(values)
    n_group_members = len(values)
    n_agree = count_agree(values)
    n_disagree = count_disagree(values)
    return float(full_group_divisive_prevalence(n_agree, n_disagree, n_group_members))


def permutation_full_group_divisive_pvalue(
    in_group_values: ArrayLike,
    out_group_values: ArrayLike,
    n_resamples: int = 999,
    random_state: Optional[int | np.random.Generator] = None,
) -> float:
    """One-sided permutation test for stronger whole-group divisiveness in the focal group."""
    in_group_values = np.asarray(in_group_values)
    out_group_values = np.asarray(out_group_values)

    if len(in_group_values) == 0 or len(out_group_values) == 0 or n_resamples <= 0:
        return 1.0

    observed = (
        _full_group_divisive_statistic(in_group_values)
        - _full_group_divisive_statistic(out_group_values)
    )

    if observed <= 0:
        return 1.0

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    result = permutation_test(
        data=(in_group_values, out_group_values),
        statistic=lambda x, y: (
            _full_group_divisive_statistic(x)
            - _full_group_divisive_statistic(y)
        ),
        permutation_type="independent",
        vectorized=False,
        alternative="greater",
        n_resamples=n_resamples,
        random_state=rng,
    )
    return float(result.pvalue)


def classify_signal_strength(
    *,
    selected: bool,
    effect_size: float,
    p_value: float,
    n_seen: int,
    group_size: int,
    strong_effect_min: float,
    strong_small_group_cutoff: Optional[int],
    strong_large_group_participation_min: Optional[float],
    strong_p_max: Optional[float],
) -> Literal["normal", "strong"]:
    """Classify a selected statement as having normal or strong signal.

    A statement can only be strong if it:
    - passes inferential selection
    - clears the effect-size threshold
    - clears the p-value threshold
    - reaches the required participation level

    Participation policy:
    - if ``group_size`` is smaller than ``strong_small_group_cutoff``,
      require full participation
    - otherwise require ``strong_large_group_participation_min``
    """
    if not selected or effect_size < strong_effect_min:
        return "normal"

    if strong_p_max is not None and p_value > strong_p_max:
        return "normal"

    if group_size <= 0:
        return "normal"

    participation = n_seen / group_size
    required_participation = strong_large_group_participation_min

    if (
        strong_small_group_cutoff is not None
        and group_size < strong_small_group_cutoff
    ):
        required_participation = 1.0

    if (
        required_participation is not None
        and participation < required_participation
    ):
        return "normal"

    return "strong"


def is_statement_agree_significant(row: pd.Series, confidence=0.90) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    pat, rat = [row[col] for col in ["pat", "rat"]]
    is_agreement_significant = is_significant(pat, confidence) and is_significant(
        rat, confidence
    )
    return is_agreement_significant


def is_statement_disagree_significant(row: pd.Series, confidence=0.90) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    pdt, rdt = [row[col] for col in ["pdt", "rdt"]]
    is_disagreement_significant = is_significant(pdt, confidence) and is_significant(
        rdt, confidence
    )
    return is_disagreement_significant


def is_statement_significant(row: pd.Series, confidence=0.90) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    # Require at least some agree or disagree votes (not just passes)
    if row["na"] == 0 and row["nd"] == 0:
        return False
    is_agreement_significant = is_statement_agree_significant(row, confidence)
    is_disagreement_significant = is_statement_disagree_significant(row, confidence)

    return is_agreement_significant or is_disagreement_significant


# DIVERGENCE FROM POLIS: Polis determines repful_for using the higher of
# ra (agree representativeness ratio) vs rd (disagree representativeness ratio).
# We instead use significance tests (rat/rdt z-scores) with a confidence
# threshold, falling back to the raw z-score comparison only when neither
# direction passes the significance test.
def get_statement_repful_for(
    row: pd.Series, confidence=0.90
) -> Literal["agree", "disagree"]:
    "Get if statement is significant for agree or disagree."
    has_repness = "rat" in row and "rdt" in row
    format_style = "group-repness" if has_repness else "consensus"

    if format_style == "consensus":
        pat, pdt = [row[col] for col in ["pat", "pdt"]]
        is_repful_for_agree = pat > pdt
        repful_for = "agree" if is_repful_for_agree else "disagree"
        return repful_for

    # now rat and rdt exist
    agree_sig = is_statement_agree_significant(row, confidence)
    disagree_sig = is_statement_disagree_significant(row, confidence)

    if agree_sig and disagree_sig:
        # Both directions significant — pick the stronger one by combined z-score.
        agree_strength = row["pat"] + row["rat"]
        disagree_strength = row["pdt"] + row["rdt"]
        return "agree" if agree_strength >= disagree_strength else "disagree"
    if agree_sig:
        return "agree"
    if disagree_sig:
        return "disagree"
    # Fallback: neither significant — compare raw z-scores.
    rat, rdt = row["rat"], row["rdt"]
    return "agree" if rat > rdt else "disagree"


def beats_best_by_repness_test(
    this_row: pd.Series,
    best_row: pd.Series | None,
) -> bool:
    """
    Returns True if a given comment/group stat has a more representative z-score than
    current_best_z. Used for ensuring at least one representative comment for every group,
    even if none remain after more thorough filters.
    """
    if best_row is not None:
        this_repness_test = max(this_row["rat"], this_row["rdt"])
        best_repness_test = max(best_row["rat"], best_row["rdt"])

        return this_repness_test > best_repness_test
    else:
        return True


def beats_best_of_agrees(
    this_row: pd.Series,
    best_row: pd.Series | None,
    confidence: float = 0.90,
) -> bool:
    """
    Like beats_best_by_repness_test, but only considers agrees. Additionally, doesn't
    focus solely on repness, but also on raw probability of agreement, so as to
    ensure that there is some representation of what people in the group agree on.
    """
    # Explicitly don't allow something that hasn't been voted on at all
    if this_row["na"] == 0 and this_row["nd"] == 0:
        return False

    if best_row is not None:
        if best_row["ra"] > 1.0:
            # If we have a current_best by representativeness estimate, use the more robust measurement
            repness_metric_agr = (
                lambda row: row["ra"] * row["rat"] * row["pa"] * row["pat"]
            )
            return repness_metric_agr(this_row) > repness_metric_agr(best_row)
        else:
            # If we have current_best, but only by probability estimate, just shoot for something generally agreed upon
            prob_metric = lambda row: row["pa"] * row["pat"]
            return prob_metric(this_row) > prob_metric(best_row)

    # Otherwise, accept if either repness or probability look generally good.
    # TODO: Hardcode significance at 90%, so lways one statement for group?
    return is_significant(this_row["pat"], confidence) or (
        this_row["ra"] > 1.0 and this_row["pa"] > 0.5
    )


# For making it more legible to get index of agree/disagree votes in numpy array.
votes = SimpleNamespace(A=0, D=1)


def count_votes(
    values: ArrayLike,
    vote_value: Optional[int] = None,
) -> np.int64 | NDArray[np.int64]:
    values = np.asarray(values)
    if vote_value:
        # Count votes that match value.
        return np.sum(values == vote_value, axis=0)
    else:
        # Count any non-missing values.
        return np.sum(np.isfinite(values), axis=0)


def count_disagree(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values, -1)


def count_agree(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values, 1)


def count_all_votes(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values)


def probability(count, total, pseudo_count: ArrayLike = 1):
    """Probability with Laplace smoothing"""
    return (pseudo_count + count) / (np.multiply(pseudo_count, 2) + total)


def calculate_comment_statistics(
    vote_matrix: VoteMatrix,
    cluster_labels: Optional[list[int] | NDArray[np.integer]] = None,
    pseudo_count: int = 1,
    consensus_mode: Literal["standard", "legacy"] = "standard",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Calculates comparative statement statistics across all votes and groups, using only efficient numpy operations.

    Note: when no cluster_labels are supplied, we internally apply the group `0` to each row,
    and calculated values can be accessed in the first group index.

    The representativeness metric is defined as:
    R_v(g,c) = P_v(g,c) / P_v(~g,c)

    Where:
    - P_v(g,c) is probability of vote v on comment c in group g
    - P_v(~g,c) is probability of vote v on comment c in all groups except g

    And:
    - N(g,c) is the total number of non-missing votes on comment c in group g
    - N_v(g,c) is the total number of vote v on comment c in group g

    Args:
        vote_matrix (VoteMatrix): A raw vote_matrix
        cluster_labels (Optional[list[int]]): An optional list of cluster labels to determine groups.

    Returns:
        N_g_c (np.ndarray[int]): numpy matrix with counts of non-missing votes on comments/groups
        N_v_g_c (np.ndarray[int]): numpy matrix with counts of vote types on comments/groups
        P_v_g_c (np.ndarray[float]): numpy matrix with probabilities of vote types on comments/groups
        R_v_g_c (np.ndarray[float]): numpy matrix with representativeness of vote types on comments/groups
        P_v_g_c_test (np.ndarray[float]): test z-scores for probability of votes/comments/groups
        R_v_g_c_test (np.ndarray[float]): test z-scores for representativeness of votes/comments/groups
        C_v_c (np.ndarray[float]): group-aware consensus scores for each statement.
    """
    if cluster_labels is None:
        # Make a single group if no labels supplied.
        participant_count = len(vote_matrix.index)
        cluster_labels = [0] * participant_count

    # Get the vote matrix values
    X = vote_matrix.values

    group_count = len(set(cluster_labels))
    statement_ids = vote_matrix.columns

    # Set up all the variables to be populated.
    N_g_c = np.empty([group_count, len(statement_ids)], dtype="int32")
    N_v_g_c = np.empty(
        [len(votes.__dict__), group_count, len(statement_ids)], dtype="int32"
    )
    P_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    P_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    C_v_c = np.empty([len(votes.__dict__), len(statement_ids)])

    for gid in range(group_count):
        # Create mask for the participants in target group
        in_group_mask = np.asarray(cluster_labels) == gid
        X_in_group = X[in_group_mask]

        # Count any votes [-1, 0, 1] for all statements/features at once

        # NON-GROUP STATS

        # For in-group
        n_agree_in_group = N_v_g_c[votes.A, gid, :] = count_agree(X_in_group)  # na
        n_disagree_in_group = N_v_g_c[votes.D, gid, :] = count_disagree(
            X_in_group
        )  # nd
        n_votes_in_group = N_g_c[gid, :] = count_all_votes(X_in_group)  # ns

        # Calculate probabilities
        p_agree_in_group = P_v_g_c[votes.A, gid, :] = probability(
            n_agree_in_group, n_votes_in_group, pseudo_count
        )  # pa
        p_disagree_in_group = P_v_g_c[votes.D, gid, :] = probability(
            n_disagree_in_group, n_votes_in_group, pseudo_count
        )  # pd

        # Calculate probability test z-scores
        P_v_g_c_test[votes.A, gid, :] = one_prop_test(
            n_agree_in_group, n_votes_in_group
        )  # pat
        P_v_g_c_test[votes.D, gid, :] = one_prop_test(
            n_disagree_in_group, n_votes_in_group
        )  # pdt

        # GROUP COMPARISON STATS

        out_group_mask = ~in_group_mask
        X_out_group = X[out_group_mask]

        # For out-group
        n_agree_out_group = count_agree(X_out_group)
        n_disagree_out_group = count_disagree(X_out_group)
        n_votes_out_group = count_all_votes(X_out_group)

        # Calculate out-group probabilities
        p_agree_out_group = probability(
            n_agree_out_group, n_votes_out_group, pseudo_count
        )
        p_disagree_out_group = probability(
            n_disagree_out_group, n_votes_out_group, pseudo_count
        )

        # Calculate representativeness
        R_v_g_c[votes.A, gid, :] = p_agree_in_group / p_agree_out_group  # ra
        R_v_g_c[votes.D, gid, :] = p_disagree_in_group / p_disagree_out_group  # rd

        # Calculate representativeness test z-scores
        R_v_g_c_test[votes.A, gid, :] = two_prop_test(
            n_agree_in_group, n_agree_out_group, n_votes_in_group, n_votes_out_group
        )  # rat
        R_v_g_c_test[votes.D, gid, :] = two_prop_test(
            n_disagree_in_group,
            n_disagree_out_group,
            n_votes_in_group,
            n_votes_out_group,
        )  # rdt

    # Calculate group-aware consensus
    # Reference: https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj#L615-L636
    n_groups = P_v_g_c.shape[1]

    if consensus_mode == "standard":
        # Jeffreys prior (0.5), effective agreement, geometric mean
        P_agree_c = np.empty([group_count, len(statement_ids)])
        P_disagree_c = np.empty([group_count, len(statement_ids)])
        for gid in range(group_count):
            P_agree_c[gid, :] = probability(N_v_g_c[votes.A, gid, :], N_g_c[gid, :], 0.5)
            P_disagree_c[gid, :] = probability(N_v_g_c[votes.D, gid, :], N_g_c[gid, :], 0.5)
        agree_scores = P_agree_c * (1 - P_disagree_c)
        disagree_scores = P_disagree_c * (1 - P_agree_c)
        C_v_c[votes.A, :] = agree_scores.prod(axis=0) ** (1.0 / n_groups)
        C_v_c[votes.D, :] = disagree_scores.prod(axis=0) ** (1.0 / n_groups)
    else:
        # Legacy (Polis): Laplace smoothing (reuses P_v_g_c), raw product
        C_v_c[votes.A, :] = P_v_g_c[votes.A, :, :].prod(axis=0)
        C_v_c[votes.D, :] = P_v_g_c[votes.D, :, :].prod(axis=0)

    return (
        N_g_c,  # ns
        N_v_g_c,  # na / nd
        P_v_g_c,  # pa / pd
        R_v_g_c,  # ra / rd
        P_v_g_c_test,  # pat / pdt
        R_v_g_c_test,  # rat / rdt
        C_v_c,  # gac
    )


def format_comment_stats(statement: pd.Series) -> PolisRepnessStatement:
    """
    Format internal statistics into concise agree/disagree format.
    Uses either consensus style or group-repness style depending on available fields.

    Args:
        statement (pd.Series): A dataframe row with statement stats in verbose format.

    Returns:
        PolisRepnessStatement: A dict with keys matching expected Polis format.
    """
    has_repness = "rat" in statement and "rdt" in statement
    format_style = "group-repness" if has_repness else "consensus"

    # Define the field mappings
    agree_fields = {
        "n-success": "na",
        "p-success": "pa",
        "p-test": "pat",
        "repness": "ra",
        "repness-test": "rat",
    }
    disagree_fields = {
        "n-success": "nd",
        "p-success": "pd",
        "p-test": "pdt",
        "repness": "rd",
        "repness-test": "rdt",
    }

    # Select score source
    if format_style == "group-repness":
        score_agree = float(statement["rat"])
        score_disagree = float(statement["rdt"])
    else:
        score_agree = float(statement["pat"])
        score_disagree = float(statement["pdt"])

    use_agree = score_agree > score_disagree
    fields = agree_fields if use_agree else disagree_fields
    direction = "agree" if use_agree else "disagree"

    result = {
        "tid": int(statement["statement_id"]),
        "n-success": int(statement[fields["n-success"]]),
        "n-trials": int(statement["ns"]),
        "p-success": float(statement[fields["p-success"]]),
        "p-test": float(statement[fields["p-test"]]),
    }

    if format_style == "group-repness":
        result["repness"] = float(statement[fields["repness"]])
        result["repness-test"] = float(statement[fields["repness-test"]])
        result["repful-for"] = direction
    else:
        result["cons-for"] = direction

    return result


def calculate_comment_statistics_dataframes(
    vote_matrix: VoteMatrix,
    cluster_labels: Optional[list[int] | NDArray[np.integer]] = None,
    pseudo_count: int = 1,
    consensus_mode: Literal["standard", "legacy"] = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates comparative statement statistics across all votes and groups, generating dataframes.

    This returns both group-specific statistics, and also overall stats (group-aware consensus).

    Args:
        vote_matrix (VoteMatrix): The vote matrix where rows are voters, columns are statements,
                                  and values are votes (1 for agree, -1 for disagree, 0 for pass).
        cluster_labels (np.ndarray): Array of cluster labels for each participant row in the vote matrix.
        pseudo_count (int): Smoothing parameter to avoid division by zero. Default is 1.

    Returns:
        pd.DataFrame: DataFrame (MultiIndex on group/statement) containing verbose statistics for each statement per group.
        pd.DataFrame: DataFrame containing group-aware consensus scores for each statement.
    """
    N_g_c, N_v_g_c, P_v_g_c, R_v_g_c, P_v_g_c_test, R_v_g_c_test, C_v_c = (
        calculate_comment_statistics(
            vote_matrix=vote_matrix,
            cluster_labels=cluster_labels,
            pseudo_count=pseudo_count,
            consensus_mode=consensus_mode,
        )
    )

    if cluster_labels is None:
        # Make a single group if no labels supplied.
        participant_count = len(vote_matrix.index)
        cluster_labels = [0] * participant_count

    cluster_labels_arr = np.asarray(cluster_labels, dtype=int)
    participant_count = len(cluster_labels_arr)
    group_count = len(set(cluster_labels))
    group_sizes = np.bincount(cluster_labels_arr, minlength=group_count)
    total_agree = N_v_g_c[votes.A, :, :].sum(axis=0)
    total_disagree = N_v_g_c[votes.D, :, :].sum(axis=0)
    total_seen = N_g_c.sum(axis=0)
    group_frames = []
    for group_id in range(group_count):
        na = N_v_g_c[votes.A, group_id, :]
        nd = N_v_g_c[votes.D, group_id, :]
        ns = N_g_c[group_id, :]
        group_df = pd.DataFrame(
            {
                "na": na,  # number agree votes
                "nd": nd,  # number disagree votes
                "ns": ns,  # number seen/total/non-missing votes
                "group_size": group_sizes[group_id],
                "out_group_size": participant_count - group_sizes[group_id],
                "na_out": total_agree - na,
                "nd_out": total_disagree - nd,
                "ns_out": total_seen - ns,
                "pa": P_v_g_c[votes.A, group_id, :],  # probability agree
                "pd": P_v_g_c[votes.D, group_id, :],  # probability disagree
                "pat": P_v_g_c_test[
                    votes.A, group_id, :
                ],  # probability agree test z-score
                "pdt": P_v_g_c_test[
                    votes.D, group_id, :
                ],  # probability disagree test z-score
                "ra": R_v_g_c[
                    votes.A, group_id, :
                ],  # repness of agree (representativeness)
                "rd": R_v_g_c[
                    votes.D, group_id, :
                ],  # repness of disagree (representativeness)
                "rat": R_v_g_c_test[
                    votes.A, group_id, :
                ],  # repress of agree test z-score
                "rdt": R_v_g_c_test[
                    votes.D, group_id, :
                ],  # repress of disagree test z-score
            },
            index=vote_matrix.columns,
        )
        group_df["group_id"] = group_id
        group_df["statement_id"] = vote_matrix.columns
        group_frames.append(group_df)
    # Create a MultiIndex dataframe
    grouped_stats_df = pd.concat(group_frames, ignore_index=True).set_index(
        ["group_id", "statement_id"]
    )

    group_aware_consensus_df = pd.DataFrame(
        {
            "group-aware-consensus": C_v_c[votes.A, :],
            "group-aware-consensus-agree": C_v_c[votes.A, :],
            "group-aware-consensus-disagree": C_v_c[votes.D, :]
        },
        index=vote_matrix.columns,
    )

    return grouped_stats_df, group_aware_consensus_df


def repness_metric(df: pd.DataFrame) -> pd.Series:
    metric = df["repness"] * df["repness-test"] * df["p-success"] * df["p-test"]
    return metric


# Reference: https://github.com/compdemocracy/polis/blob/fd440c3e3ca302d08ce3cca870fc39b834c96b86/math/src/polismath/math/conversation.clj#L308C1-L312C28
def importance_metric(
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    n_total: ArrayLike,
    extremity: ArrayLike,
    pseudo_count: ArrayLike = 1,
) -> np.ndarray:
    n_agree, n_disagree, n_total, extremity = map(
        np.asarray, (n_agree, n_disagree, n_total, extremity)
    )  # Ensure inputs are NumPy arrays
    n_pass = n_total - (n_agree + n_disagree)
    prob_agree = probability(n_agree, n_total, pseudo_count)
    prob_pass = probability(n_pass, n_total, pseudo_count)
    # From in the academic paper:
    # importance = prob_agree * (1 - prob_pass) * (1 + extremity)
    prob_engagement = 1 - prob_pass
    # This is what happens:
    #   - total agreement scales by 1 (disagreement down-scales)
    #   - total engagement (agree or disagree) scales by 1 (passing down-scales)
    #   - if a virtual participant who only votes agree on this statement
    #     does not move from the center, this statement scales by 1.
    #     The more they would travel from the center, the more the statement up-scales.
    #     (The more a statement contributes to principal components, the more upscaling.)
    importance = prob_agree * prob_engagement * (1 + extremity)
    return importance


# Reference: https://github.com/compdemocracy/polis/blob/fd440c3e3ca302d08ce3cca870fc39b834c96b86/math/src/polismath/math/conversation.clj#L318-L327
def priority_metric(
    is_meta: ArrayLike,
    n_agree: ArrayLike,
    n_disagree: ArrayLike,
    n_total: ArrayLike,
    extremity: ArrayLike,
    # This might need tuning.
    meta_priority: int = 7,
    pseudo_count: ArrayLike = 1,
) -> np.ndarray:
    """
    Calculate comment priority metric for any single statement of lists of statements.

    Args:
        is_meta (bool | list[bool]): Whether statement is marked as metadata
        n_agree (int | list[int]): Number of agree votes
        n_disagree (int | list[int]): Number of disagree votes
        n_total (int | list[int]): Number of total votes (agree/disagree/pass)
        extremity (float | list[float]): Euclidean distance of a single-voting participant from origin
        meta_priority (int): Unbiased (pre-squared) priority metric used for meta statements
        pseudo_count (int | list[int]): pseudo-count value for Laplace smoothing
    Returns:
        float | np.ndarray[float]: A priority metric score between 0 and infinity.
    """
    # Ensure inputs are NumPy arrays
    n_agree, n_disagree, n_total, extremity = map(
        np.asarray, (n_agree, n_disagree, n_total, extremity)
    )
    importance = importance_metric(
        n_agree, n_disagree, n_total, extremity, pseudo_count
    )
    # Form in the academic paper: (transformed)
    # newness_scale_factor = 1 + 2**(3 - (n_total/5)))
    # newness_scale_factor = 1 + 2**3 * (2**(-(n_total/5)))
    # newness_scale_factor = 1 + 8    * (2**(-(n_total/5)))
    NO_VOTES_SCALE_FACTOR = 9
    newness_scale_factor = 1 + (NO_VOTES_SCALE_FACTOR - 1) * np.power(2, -(n_total / 5))
    priority = importance * newness_scale_factor

    # Assign meta priority where is_meta is True
    priority = np.where(is_meta, meta_priority, priority)

    boosted_bias_priority = np.power(priority, 2)
    return boosted_bias_priority


# Figuring out select-rep-comments flow
# See: https://github.com/compdemocracy/polis/blob/7bf9eccc287586e51d96fdf519ae6da98e0f4a70/math/src/polismath/math/repness.clj#L209C7-L209C26
# TODO: omg please clean this up.
def select_representative_statements(
    grouped_stats_df: pd.DataFrame,
    mod_out_statement_ids: list[int] = [],
    pick_max: int = 5,
    confidence: float = 0.90,
) -> PolisRepness:
    """
    Selects statistically representative statements from each group cluster.

    This is expected to match the Polis outputs when all defaults are set.

    Args:
        grouped_stats_df (pd.DataFrame): MultiIndex Dataframe of statement statistics, indexed by group and statement.
        mod_out_statement_ids (list[int]): A list of statements to ignore from selection algorithm
        pick_max (int): Max number of statements selected per group
        confidence (float): Percent confidence interval (in decimal), within which selected statements are deemed significant

    Returns:
        PolisRepness: A dict object with lists of statements keyed to groups, matching Polis format.
    """
    repness = {}
    # TODO: Should this be done elsewhere? A column in MultiIndex dataframe?
    mod_out_mask = grouped_stats_df.index.get_level_values("statement_id").isin(
        mod_out_statement_ids
    )
    grouped_stats_df = grouped_stats_df[~mod_out_mask]  # type: ignore
    for gid, group_df in grouped_stats_df.groupby(level="group_id"):
        # Bring statement_id into regular column.
        group_df = group_df.reset_index()

        best_agree = None
        # Track the best-agree, to bring to top if exists.
        for _, row in group_df.iterrows():
            if beats_best_of_agrees(row, best_agree, confidence):
                best_agree = row

        sig_filter = lambda row: is_statement_significant(row, confidence)
        sufficient_statements_row_mask = group_df.apply(sig_filter, axis="columns")
        sufficient_statements = group_df[sufficient_statements_row_mask]

        # Track the best, even if doesn't meet sufficient minimum, to have at least one.
        best_overall = None
        if len(sufficient_statements) == 0:
            for _, row in group_df.iterrows():
                if beats_best_by_repness_test(row, best_overall):
                    best_overall = row
        else:
            # Finalize statements into output format.
            # TODO: Figure out how to finalize only at end in output. Change repness_metric?
            sufficient_statements = (
                pd.DataFrame(
                    [
                        format_comment_stats(row)
                        for _, row in sufficient_statements.iterrows()
                    ]
                )
                # Create a column to sort repnress, then remove.
                .assign(repness_metric=repness_metric)
                .sort_values(by="repness_metric", ascending=False)
                .drop(columns="repness_metric")
            )

        if best_agree is not None:
            best_agree = format_comment_stats(best_agree)
            best_agree.update({"n-agree": best_agree["n-success"], "best-agree": True})
            best_head = [best_agree]
        elif best_overall is not None:
            best_overall = format_comment_stats(best_overall)
            best_head = [best_overall]
        else:
            best_head = []

        selected = best_head
        selected = selected + [
            row.to_dict()
            for _, row in sufficient_statements.iterrows()
            if best_head
            # Skip any statements already in best_head
            and best_head[0]["tid"] != row["tid"]
        ]
        selected = selected[:pick_max]
        # Does the work of agrees-before-disagrees sort in polismath, since "a" before "d".
        selected = sorted(selected, key=lambda row: row["repful-for"])
        repness[gid] = selected

    return repness  # type:ignore


def rank_representative_statements(
    grouped_stats_df: pd.DataFrame,
    mod_out_statement_ids: list[int] = [],
    fdr_rate: float = 0.10,
    vote_matrix: Optional[VoteMatrix] = None,
    cluster_labels: Optional[list[int] | NDArray[np.integer]] = None,
    divisive_n_resamples: int = 999,
    divisive_random_state: Optional[int] = None,
    strong_effect_min: float = 1.0,
    strong_small_group_cutoff: Optional[int] = 5,
    strong_large_group_participation_min: Optional[float] = 0.8,
    strong_p_max: Optional[float] = 0.05,
) -> dict[int, list[RankedRepnessStatement]]:
    """
    Rank representative statements for Agora using thresholded three-label selection.

    Args:
        grouped_stats_df (pd.DataFrame): MultiIndex DataFrame of statement statistics, indexed by group and statement.
        mod_out_statement_ids (list[int]): A list of statements to ignore.
        fdr_rate (float): False discovery rate for BH procedure.
        vote_matrix (VoteMatrix): Clustered vote matrix aligned with ``cluster_labels``.
        cluster_labels (list[int] | np.ndarray[int]): Cluster labels aligned to ``vote_matrix`` rows.

    Returns:
        A dict mapping group_id to a list of RankedRepnessStatement, sorted by effect_size descending.
    """
    if vote_matrix is None or cluster_labels is None:
        raise ValueError(
            "vote_matrix and cluster_labels are required for Agora representative ranking"
        )

    mod_out_mask = grouped_stats_df.index.get_level_values("statement_id").isin(
        mod_out_statement_ids
    )
    grouped_stats_df = grouped_stats_df[~mod_out_mask]  # type: ignore

    cluster_labels_arr = np.asarray(cluster_labels, dtype=int)
    if len(cluster_labels_arr) != len(vote_matrix.index):
        raise ValueError("cluster_labels must align with vote_matrix rows")

    n_groups = np.unique(cluster_labels_arr).size
    if n_groups < 2:
        raise ValueError(
            f"Agora representative ranking requires at least 2 distinct groups; got {n_groups}."
        )

    result: dict[int, list[RankedRepnessStatement]] = {}
    group_rng = np.random.default_rng(divisive_random_state)

    for gid, group_df in grouped_stats_df.groupby(level="group_id"):
        group_df = group_df.reset_index()

        # Per-direction p-values: probability test and representativeness test.
        p_prob_a = z_to_pvalue(group_df["pat"].values)
        p_rep_a = z_to_pvalue(group_df["rat"].values)
        p_prob_d = z_to_pvalue(group_df["pdt"].values)
        p_rep_d = z_to_pvalue(group_df["rdt"].values)

        # Combine probability + representativeness p-values using Simes' method,
        # which is valid under positive dependence (both tests use the same votes).
        p_simes_a = simes_combine(p_prob_a, p_rep_a)
        p_simes_d = simes_combine(p_prob_d, p_rep_d)

        group_mask = cluster_labels_arr == gid
        out_group_mask = ~group_mask
        group_vote_matrix = vote_matrix.loc[group_mask, :]
        out_group_vote_matrix = vote_matrix.loc[out_group_mask, :]

        agree_prevalence_in = full_group_prevalence(
            group_df["na"].values,
            group_df["group_size"].values,
        )
        disagree_prevalence_in = full_group_prevalence(
            group_df["nd"].values,
            group_df["group_size"].values,
        )
        agree_prevalence_out = full_group_prevalence(
            group_df["na_out"].values,
            group_df["out_group_size"].values,
        )
        disagree_prevalence_out = full_group_prevalence(
            group_df["nd_out"].values,
            group_df["out_group_size"].values,
        )

        effect_agree = agora_thresholded_effect(
            prevalence_in=agree_prevalence_in,
            prevalence_out=agree_prevalence_out,
            out_group_size=group_df["out_group_size"].values,
            min_out_count=1.0,
        )
        effect_disagree = agora_thresholded_effect(
            prevalence_in=disagree_prevalence_in,
            prevalence_out=disagree_prevalence_out,
            out_group_size=group_df["out_group_size"].values,
            min_out_count=1.0,
        )
        effect_divisive = agora_thresholded_divisive_effect(
            n_agree_in=group_df["na"].values,
            n_disagree_in=group_df["nd"].values,
            group_size_in=group_df["group_size"].values,
            n_agree_out=group_df["na_out"].values,
            n_disagree_out=group_df["nd_out"].values,
            group_size_out=group_df["out_group_size"].values,
        )
        divisiveness_values = full_group_divisive_prevalence(
            group_df["na"].values,
            group_df["nd"].values,
            group_df["group_size"].values,
        )

        p_divisive = np.ones(len(group_df), dtype=float)
        for idx, row in group_df.iterrows():
            statement_id = row["statement_id"]
            in_group_values = group_vote_matrix.loc[:, statement_id].values
            out_group_values = out_group_vote_matrix.loc[:, statement_id].values
            p_divisive[idx] = permutation_full_group_divisive_pvalue(
                in_group_values=in_group_values,
                out_group_values=out_group_values,
                n_resamples=divisive_n_resamples,
                random_state=group_rng,
            )

        label_effects = np.column_stack([effect_agree, effect_disagree, effect_divisive])
        label_p_values = np.column_stack([p_simes_a, p_simes_d, p_divisive])
        candidate_mask = agora_label_candidate_mask(
            n_agree=group_df["na"].values,
            n_disagree=group_df["nd"].values,
            n_seen=group_df["ns"].values,
        )
        fallback_local_scores = agora_local_label_scores(
            n_agree=group_df["na"].values,
            n_disagree=group_df["nd"].values,
            n_seen=group_df["ns"].values,
        )
        fallback_local_scores[
            np.minimum(group_df["na"].values, group_df["nd"].values) < 2,
            2,
        ] = -np.inf
        winning_labels = choose_agora_thresholded_winners(
            label_effects=label_effects,
            label_p_values=label_p_values,
            candidate_mask=candidate_mask,
            fallback_local_scores=fallback_local_scores,
        )
        direction_names = np.array(["agree", "disagree", "divisive"], dtype=object)
        directions = direction_names[winning_labels]
        effect_sizes = label_effects[np.arange(len(group_df)), winning_labels]
        p_combined = label_p_values[np.arange(len(group_df)), winning_labels]

        has_votes = (group_df["na"].values > 0) | (group_df["nd"].values > 0)
        selected_mask, adjusted = apply_bh_with_vote_filter(
            p_combined, has_votes, fdr_rate
        )

        n = len(p_combined)
        rank_order = np.argsort(-effect_sizes)
        ranks = np.empty(n, dtype=int)
        ranks[rank_order] = np.arange(1, n + 1)
        statements: list[RankedRepnessStatement] = []
        for idx in rank_order:
            row = group_df.iloc[idx]
            n_seen = int(row["ns"])
            group_size = int(row["group_size"])
            signal_strength = classify_signal_strength(
                selected=bool(selected_mask[idx]),
                effect_size=float(effect_sizes[idx]),
                p_value=float(p_combined[idx]),
                n_seen=n_seen,
                group_size=group_size,
                strong_effect_min=strong_effect_min,
                strong_small_group_cutoff=strong_small_group_cutoff,
                strong_large_group_participation_min=(
                    strong_large_group_participation_min
                ),
                strong_p_max=strong_p_max,
            )
            statements.append(
                RankedRepnessStatement(
                    statement_id=int(row["statement_id"]),
                    repful_for=directions[idx],
                    signal_strength=signal_strength,
                    na=int(row["na"]),
                    nd=int(row["nd"]),
                    ns=int(row["ns"]),
                    group_size=int(row["group_size"]),
                    out_group_size=int(row["out_group_size"]),
                    na_out=int(row["na_out"]),
                    nd_out=int(row["nd_out"]),
                    ns_out=int(row["ns_out"]),
                    pa=float(row["pa"]),
                    pd=float(row["pd"]),
                    pat=float(row["pat"]),
                    pdt=float(row["pdt"]),
                    ra=float(row["ra"]),
                    rd=float(row["rd"]),
                    rat=float(row["rat"]),
                    rdt=float(row["rdt"]),
                    divisiveness=float(divisiveness_values[idx]),
                    agree_effect=float(effect_agree[idx]),
                    disagree_effect=float(effect_disagree[idx]),
                    divisive_effect=float(effect_divisive[idx]),
                    agree_p_value=float(p_simes_a[idx]),
                    disagree_p_value=float(p_simes_d[idx]),
                    divisive_p_value=float(p_divisive[idx]),
                    effect_size=float(effect_sizes[idx]),
                    p_value=float(p_combined[idx]),
                    adjusted_p_value=float(adjusted[idx]),
                    selected=bool(selected_mask[idx]),
                    rank=int(ranks[idx]),
                )
            )

        result[int(gid)] = statements

    return result


def populate_priority_calculations_into_statements_df(
    statements_df: pd.DataFrame,
    vote_matrix: VoteMatrix,
) -> pd.DataFrame:
    statements_df = statements_df.copy()
    statements_df["extremity"] = calculate_extremity(
        statements_df.loc[:, ["x", "y"]].transpose()
    )

    n_agree = (vote_matrix == 1).sum(axis=0)
    n_disagree = (vote_matrix == -1).sum(axis=0)
    n_total = vote_matrix.notna().sum(axis=0)

    statements_df["n_agree"] = n_agree.reindex(statements_df.index).astype("Int64")
    statements_df["n_disagree"] = n_disagree.reindex(statements_df.index).astype(
        "Int64"
    )
    statements_df["n_total"] = n_total.reindex(statements_df.index).astype("Int64")

    statements_df["priority"] = priority_metric(
        is_meta=statements_df["is_meta"],
        n_agree=statements_df["n_agree"],
        n_disagree=statements_df["n_disagree"],
        n_total=statements_df["n_total"],
        extremity=statements_df["extremity"],
    )
    return statements_df
