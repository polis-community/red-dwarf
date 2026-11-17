import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Optional, Literal
from types import SimpleNamespace
from scipy.stats import norm
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
            # Preserve upstream's product, including positive scores from two negatives.
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

    group_count = len(set(cluster_labels))
    group_frames = []
    for group_id in range(group_count):
        group_df = pd.DataFrame(
            {
                "na": N_v_g_c[votes.A, group_id, :],  # number agree votes
                "nd": N_v_g_c[votes.D, group_id, :],  # number disagree votes
                "ns": N_g_c[group_id, :],  # number seen/total/non-missing votes
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

    Preserves Polis's legacy best-agree and fallback behavior: a best-agree
    headline may be formatted as disagree, and fallback may select a statement
    without directional votes. These quirks are characterized in
    tests/utils/test_representative_statements.py, not corrected in this selector.

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
            n_agree = int(best_agree["na"])
            best_agree = format_comment_stats(best_agree)
            best_agree.update({"n-agree": n_agree, "best-agree": True})
            best_head = [best_agree]
        elif best_overall is not None:
            best_overall = format_comment_stats(best_overall)
            best_head = [best_overall]
        else:
            best_head = []

        head_statement_ids = {statement["tid"] for statement in best_head}
        selected = best_head + [
            row.to_dict()
            for _, row in sufficient_statements.iterrows()
            if row["tid"] not in head_statement_ids
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
) -> dict[int, list[RankedRepnessStatement]]:
    """
    Ranks ALL statements per group by effect size and marks selections via Benjamini-Hochberg.

    Args:
        grouped_stats_df (pd.DataFrame): MultiIndex DataFrame of statement statistics, indexed by group and statement.
        mod_out_statement_ids (list[int]): A list of statements to ignore.
        fdr_rate (float): False discovery rate for BH procedure.

    Returns:
        A dict mapping group_id to a list of RankedRepnessStatement, sorted by effect_size descending.
    """
    mod_out_mask = grouped_stats_df.index.get_level_values("statement_id").isin(
        mod_out_statement_ids
    )
    grouped_stats_df = grouped_stats_df[~mod_out_mask]  # type: ignore

    result: dict[int, list[RankedRepnessStatement]] = {}

    for gid, group_df in grouped_stats_df.groupby(level="group_id"):
        group_df = group_df.reset_index()

        # Per-direction p-values: probability test and representativeness test.
        p_prob_a = z_to_pvalue(group_df["pat"].values)
        p_rep_a = z_to_pvalue(group_df["rat"].values)
        p_prob_d = z_to_pvalue(group_df["pdt"].values)
        p_rep_d = z_to_pvalue(group_df["rdt"].values)

        # Combine probability + representativeness p-values using Simes' method,
        # which is valid under positive dependence (both tests use the same votes).
        # For 2 tests: p_simes = min(2 * p_min, p_max).
        p_simes_a = np.minimum(
            2 * np.minimum(p_prob_a, p_rep_a), np.maximum(p_prob_a, p_rep_a)
        )
        p_simes_d = np.minimum(
            2 * np.minimum(p_prob_d, p_rep_d), np.maximum(p_prob_d, p_rep_d)
        )

        # Pick the better direction (lower combined p-value).
        agree_better = p_simes_a <= p_simes_d
        p_combined = np.where(agree_better, p_simes_a, p_simes_d)
        directions = np.where(agree_better, "agree", "disagree")

        # Effect size: ra*pa for agree, rd*pd for disagree.
        effect_agree = group_df["ra"].values * group_df["pa"].values
        effect_disagree = group_df["rd"].values * group_df["pd"].values
        effect_sizes = np.where(agree_better, effect_agree, effect_disagree)

        # Apply BH with zero-vote filter.
        has_votes = (group_df["na"].values > 0) | (group_df["nd"].values > 0)
        selected_mask, adjusted = apply_bh_with_vote_filter(
            p_combined, has_votes, fdr_rate
        )

        # Rank by effect size descending (1-based).
        n = len(p_combined)
        rank_order = np.argsort(-effect_sizes)
        ranks = np.empty(n, dtype=int)
        ranks[rank_order] = np.arange(1, n + 1)

        # Build RankedRepnessStatement list, sorted by rank.
        statements: list[RankedRepnessStatement] = []
        for idx in rank_order:
            row = group_df.iloc[idx]
            statements.append(
                RankedRepnessStatement(
                    statement_id=int(row["statement_id"]),
                    repful_for=directions[idx],
                    na=int(row["na"]),
                    nd=int(row["nd"]),
                    ns=int(row["ns"]),
                    pa=float(row["pa"]),
                    pd=float(row["pd"]),
                    pat=float(row["pat"]),
                    pdt=float(row["pdt"]),
                    ra=float(row["ra"]),
                    rd=float(row["rd"]),
                    rat=float(row["rat"]),
                    rdt=float(row["rdt"]),
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
