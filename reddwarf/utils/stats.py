import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Optional
from types import SimpleNamespace
from scipy.stats import norm
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.types.polis import (
    PolisRepness,
    PolisRepnessStatement,
)
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
    return z_val > critical_value


def is_statement_significant(row: pd.Series, confidence=0.90) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    pat, rat, pdt, rdt = [row[col] for col in ["pat", "rat", "pdt", "rdt"]]
    is_agreement_significant = is_significant(pat, confidence) and is_significant(
        rat, confidence
    )
    is_disagreement_significant = is_significant(pdt, confidence) and is_significant(
        rdt, confidence
    )

    return is_agreement_significant or is_disagreement_significant


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
    # For each statement, multiply probabilities across groups (aka the first axis=0)
    # Reference: https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj#L615-L636
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

    sid = statement["statement_id"]

    if isinstance(sid, int):
        tid = int(sid)
    elif isinstance(sid, str) and sid.endswith("p"):
        tid = str(sid)
    else:
        tid = int(sid)

    result = {
        "tid": tid,
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
