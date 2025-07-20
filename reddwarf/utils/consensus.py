import pandas as pd
from typing import List, TypedDict
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.utils.stats import (
    calculate_comment_statistics,
    format_comment_stats,
    is_significant,
    votes,
)


class ConsensusStatement(TypedDict):
    """
    A dictionary representing a consensus statement with its statistics.
    """

    tid: int
    n_success: int
    n_trials: int
    p_success: float
    p_test: float
    cons_for: str


class ConsensusResult(TypedDict):
    """
    A dictionary containing lists of consensus statements for agreement and disagreement.
    """

    agree: List[ConsensusStatement]
    disagree: List[ConsensusStatement]


# TODO: Allow setting `pick_max`, `prob_threshold`, or `confidence`` to `None` in
# order to avoid filtering by that measure.
def select_consensus_statements(
    vote_matrix: VoteMatrix,
    mod_out_statement_ids: list[int | str] = [],
    pick_max=5,
    prob_threshold: float = 0.5,
    confidence: float = 0.9,
) -> ConsensusResult:
    """
    Select consensus statements from a given vote matrix.

    Args:
        vote_matrix (VoteMatrix): The full raw vote matrix (not just clusterable participants)
        mod_out_statement_ids (Optional[list[int]]): Statements to ignore from consensus statement selection
        pick_max (int): Max number of statements selected per agree/disagree direction
        prob_threshold (float): The cutoff probability below which statements won't be considered for consensus
        confidence (float): Percent confidence interval (in decimal), within which selected statements are deemed significant

    Returns:
        A dict of agree/disagree formatted statement dicts.
    """
    N_g_c, N_v_g_c, P_v_g_c, _, P_v_g_c_test, *_ = calculate_comment_statistics(
        vote_matrix=vote_matrix,
        cluster_labels=None,
    )
    # When no labels provided above, mock group is used.
    MOCK_GID = 0
    df = pd.DataFrame(
        {
            "na": N_v_g_c[votes.A, MOCK_GID, :],
            "nd": N_v_g_c[votes.D, MOCK_GID, :],
            "ns": N_g_c[MOCK_GID, :],
            "pa": P_v_g_c[votes.A, MOCK_GID, :],
            "pd": P_v_g_c[votes.D, MOCK_GID, :],
            "pat": P_v_g_c_test[votes.A, MOCK_GID, :],
            "pdt": P_v_g_c_test[votes.D, MOCK_GID, :],
            # agree metric = pa * pat
            "am": P_v_g_c[votes.A, MOCK_GID, :] * P_v_g_c_test[votes.A, MOCK_GID, :],
            # disagree metric = pd * pdt
            "dm": P_v_g_c[votes.D, MOCK_GID, :] * P_v_g_c_test[votes.D, MOCK_GID, :],
        },
        index=vote_matrix.columns,
    )

    # Optional filtering for mod_out_statement_ids
    if mod_out_statement_ids:
        df = df[~df.index.isin(mod_out_statement_ids)]

    # Agree candidates: pa > threshold and significant
    agree_mask = (df["pa"] > prob_threshold) & df["pat"].apply(
        lambda x: is_significant(x, confidence)
    )
    agree_candidates = df[agree_mask].copy()
    agree_candidates["consensus_agree_rank"] = (
        (-agree_candidates["am"]).rank(method="dense").astype("Int64")
    )

    # Disagree candidates: pd > threshold and significant
    disagree_mask = (df["pd"] > prob_threshold) & df["pdt"].apply(
        lambda x: is_significant(x, confidence)
    )
    disagree_candidates = df[disagree_mask].copy()
    disagree_candidates["consensus_disagree_rank"] = (
        (-disagree_candidates["dm"]).rank(method="dense").astype("Int64")
    )

    # Merge ranks back into full df
    df["consensus_agree_rank"] = agree_candidates["consensus_agree_rank"]
    df["consensus_disagree_rank"] = disagree_candidates["consensus_disagree_rank"]

    # Select top N agree/disagree statements
    if agree_candidates.empty:
        top_agree = []
    else:
        top_agree = [
            # Drop the cons-for key from final output.
            {k: v for k, v in st.items() if k != "cons-for"}
            for st in agree_candidates.sort_values("consensus_agree_rank")
            .head(pick_max)
            .reset_index()
            .apply(format_comment_stats, axis=1)
        ]

    if disagree_candidates.empty:
        top_disagree = []
    else:
        top_disagree = [
            # Drop the cons-for key from final output.
            {k: v for k, v in st.items() if k != "cons-for"}
            for st in disagree_candidates.sort_values("consensus_disagree_rank")
            .head(pick_max)
            .reset_index()
            .apply(format_comment_stats, axis=1)
        ]

    return {
        "agree": top_agree,
        "disagree": top_disagree,
    }
