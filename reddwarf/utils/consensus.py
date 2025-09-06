import pandas as pd
from typing import List, TypedDict, Dict, Any, Optional
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


class GroupedConsensusResult(TypedDict):
    """A dictionary containing consensus statements for a specific group."""
    group_id: int
    group_name: str
    agree: List[ConsensusStatement]
    disagree: List[ConsensusStatement]


# TODO: Allow setting `pick_max`, `prob_threshold`, or `confidence`` to `None` in
# order to avoid filtering by that measure.
def select_consensus_statements(
    vote_matrix: VoteMatrix,
    mod_out_statement_ids: list[int] = None,
    pick_max: int = None,
    max_limit: int = 100,
    prob_threshold: float = 0.5,
    confidence: float = 0.9,
    group_id: int = 0,
) -> ConsensusResult:
    """
    Select consensus statements from a given vote matrix for a specific group.

    Args:
        vote_matrix (VoteMatrix): The full raw vote matrix (not just clusterable participants)
        mod_out_statement_ids (Optional[list[int]]): Statements to ignore from consensus statement selection
        pick_max (int): Max number of statements selected per agree/disagree direction (default: 5)
        max_limit (int): Maximum allowed value for pick_max (default: 100)
        prob_threshold (float): The cutoff probability below which statements won't be considered for consensus
        confidence (float): Percent confidence interval (in decimal), within which selected statements are deemed significant
        group_id (int): The group ID to analyze (default: 0)

    Returns:
        A dict of agree/disagree formatted statement dicts.
    """
    N_g_c, N_v_g_c, P_v_g_c, _, P_v_g_c_test, *_ = calculate_comment_statistics(
        vote_matrix=vote_matrix,
        cluster_labels=None,
    )
    # Use the specified group_id
    df = pd.DataFrame(
        {
            "statement_id": vote_matrix.columns,  
            "na": N_v_g_c[votes.A, group_id, :],
            "nd": N_v_g_c[votes.D, group_id, :],
            "ns": N_g_c[group_id, :],
            "pa": P_v_g_c[votes.A, group_id, :],
            "pd": P_v_g_c[votes.D, group_id, :],
            "pat": P_v_g_c_test[votes.A, group_id, :],
            "pdt": P_v_g_c_test[votes.D, group_id, :],
            # agree metric = pa * pat
            "am": P_v_g_c[votes.A, group_id, :] * P_v_g_c_test[votes.A, group_id, :],
            # disagree metric = pd * pdt
            "dm": P_v_g_c[votes.D, group_id, :] * P_v_g_c_test[votes.D, group_id, :],
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

    # Apply max_limit to pick_max
    if pick_max is None or pick_max > max_limit:
        pick_max = max_limit
        
    # Select top N agree/disagree statements
    if agree_candidates.empty:
        top_agree = []
    else:
        top_agree = [
            # Drop the cons-for key from final output.
            {k: v for k, v in st.items() if k != "cons-for"}
            for st in (agree_candidates.sort_values("consensus_agree_rank")
                     .head(pick_max)
                     .reset_index()
                     .apply(format_comment_stats, axis=1))
        ]

    if disagree_candidates.empty:
        top_disagree = []
    else:
        top_disagree = [
            # Drop the cons-for key from final output.
            {k: v for k, v in st.items() if k != "cons-for"}
            for st in (disagree_candidates.sort_values("consensus_disagree_rank")
                     .head(pick_max)
                     .reset_index()
                     .apply(format_comment_stats, axis=1))
        ]

    return {
        "agree": top_agree,
        "disagree": top_disagree,
    }


def select_grouped_consensus(
    vote_matrix: VoteMatrix,
    group_ids: Optional[List[int]] = None,
    group_names: Optional[Dict[int, str]] = None,
    **kwargs
) -> List[GroupedConsensusResult]:
    """
    Select consensus statements for multiple groups.

    Args:
        vote_matrix (VoteMatrix): The full raw vote matrix
        group_ids (Optional[List[int]]): List of group IDs to analyze. If None, uses all available groups.
        group_names (Optional[Dict[int, str]]): Mapping of group IDs to group names.
                                              If None, default names will be used.
        **kwargs: Additional arguments to pass to select_consensus_statements

    Returns:
        List[GroupedConsensusResult]: A list of consensus results for each group
    """
    if group_ids is None:
        # If no group_ids provided, use all available groups
        n_groups = vote_matrix.n_groups if hasattr(vote_matrix, 'n_groups') else 1
        group_ids = list(range(n_groups))
    
    if group_names is None:
        group_names = {}
    
    results = []
    
    for group_id in group_ids:
        # Get group name or use default
        group_name = group_names.get(group_id, f"Group {group_id}")
        
        # Get consensus for this group
        consensus = select_consensus_statements(
            vote_matrix=vote_matrix,
            group_id=group_id,
            **kwargs
        )
        
        # Add to results
        results.append({
            'group_id': group_id,
            'group_name': group_name,
            'agree': consensus['agree'],
            'disagree': consensus['disagree']
        })
    
    return results
