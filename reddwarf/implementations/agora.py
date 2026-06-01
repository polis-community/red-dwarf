from typing import Optional
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame

from reddwarf.implementations import base
from reddwarf.types.agora import RankedRepnessStatement, RankedConsensusResult
from reddwarf.utils.consensus import rank_consensus_statements
from reddwarf.utils.reducer.base import ReducerModel
from reddwarf.utils.clusterer.base import ClustererModel
from reddwarf.utils.stats import rank_representative_statements


@dataclass
class AgoraClusteringResult:
    """
    Attributes:
        raw_vote_matrix (DataFrame): Raw sparse vote matrix before any processing.
        filtered_vote_matrix (DataFrame): Raw sparse vote matrix with moderated statements zero'd out.
        reducer (ReducerModel): scikit-learn reducer model fitted to vote matrix.
        clusterer (ClustererModel): scikit-learn clusterer model, fitted to participant projections.
        group_comment_stats (DataFrame): A multi-index dataframe for each statement, indexed by group ID and statement.
        statements_df (DataFrame): A dataframe with all intermediary and final statement data/calculations/metadata.
        participants_df (DataFrame): A dataframe with all intermediary and final participant data/calculations/metadata.
        participant_projections (dict): A dict of participant projected coordinates, keyed to participant ID.
        statement_projections (Optional[dict]): A dict of statement projected coordinates, keyed to statement ID.
        group_aware_consensus (dict): A nested dict of statement group-aware-consensus values.
        ranked_repness (dict[int, list[RankedRepnessStatement]]): All statements per group, ranked by effect size with BH selection.
        ranked_consensus (RankedConsensusResult): All statements ranked for consensus, with BH selection.
    """

    raw_vote_matrix: DataFrame
    filtered_vote_matrix: DataFrame
    reducer: ReducerModel
    clusterer: ClustererModel | None
    group_comment_stats: DataFrame
    statements_df: DataFrame
    participants_df: DataFrame
    participant_projections: dict
    statement_projections: Optional[dict]
    group_aware_consensus: dict
    ranked_repness: dict[int, list[RankedRepnessStatement]]
    ranked_consensus: RankedConsensusResult


def compute_effective_agreement_gac(
    grouped_stats_df: pd.DataFrame,
    statement_ids,
    n_groups: int,
) -> dict:
    """Compute GAC using effective agreement: prod(pa * (1-pd))^(1/n_groups).

    Unlike the Polis formula (raw pa product), this penalizes groups that are
    genuinely divided (high agree AND high disagree) by discounting each group's
    agreement by its disagreement.

    Args:
        grouped_stats_df: MultiIndex DataFrame with (group_id, statement_id) index, containing 'pa' and 'pd' columns.
        statement_ids: Statement IDs to compute GAC for.
        n_groups: Number of groups.

    Returns:
        Dict with 'agree' and 'disagree' keys, each mapping statement_id to GAC score.
    """
    agree = {}
    disagree = {}
    for sid in statement_ids:
        ea_prod = 1.0
        ed_prod = 1.0
        for gid in range(n_groups):
            pa = grouped_stats_df.loc[(gid, sid), "pa"]
            pd_val = grouped_stats_df.loc[(gid, sid), "pd"]
            ea_prod *= pa * (1 - pd_val)
            ed_prod *= pd_val * (1 - pa)
        agree[sid] = ea_prod ** (1.0 / n_groups)
        disagree[sid] = ed_prod ** (1.0 / n_groups)
    return {"agree": agree, "disagree": disagree}


def run_pipeline(
    fdr_rate: float = 0.10,
    divisive_n_resamples: int = 999,
    divisive_random_state: Optional[int] = None,
    strong_effect_min: float = 1.0,
    strong_small_group_cutoff: Optional[int] = 5,
    strong_large_group_participation_min: Optional[float] = 0.8,
    strong_p_max: Optional[float] = 0.05,
    **kwargs,
) -> AgoraClusteringResult:
    """
    Agora clustering pipeline. Runs the base pipeline and adds ranked
    representative/consensus statements with Benjamini-Hochberg selection.

    Accepts all the same arguments as base.run_pipeline(), plus:

    Args:
        fdr_rate (float): False discovery rate for Benjamini-Hochberg selection.
        divisive_n_resamples (int): Number of permutation resamples for divisive p-values.
        divisive_random_state (Optional[int]): RNG seed for divisive permutation tests.
        strong_effect_min (float): Minimum effect size for a selected statement to be labeled strong.
        strong_small_group_cutoff (Optional[int]): Groups smaller than this require full participation for a statement to be labeled strong.
        strong_large_group_participation_min (Optional[float]): Participation rate required for groups at or above the small-group cutoff.
        strong_p_max (Optional[float]): Maximum p-value for strong labeling.
        **kwargs: All arguments forwarded to base.run_pipeline().

    Returns:
        AgoraClusteringResult: Clustering results with ranked statement outputs.
    """
    base_result = base.run_pipeline(**kwargs)

    ranked_repness = rank_representative_statements(
        grouped_stats_df=base_result.group_comment_stats,
        vote_matrix=base_result.raw_vote_matrix.loc[
            base_result.participants_df[base_result.participants_df["to_cluster"]].index,
            :,
        ],
        cluster_labels=base_result.participants_df.loc[
            base_result.participants_df["to_cluster"], "cluster_id"
        ].astype(int).tolist(),
        mod_out_statement_ids=kwargs.get("mod_out_statement_ids", []),
        fdr_rate=fdr_rate,
        divisive_n_resamples=divisive_n_resamples,
        divisive_random_state=divisive_random_state,
        strong_effect_min=strong_effect_min,
        strong_small_group_cutoff=strong_small_group_cutoff,
        strong_large_group_participation_min=strong_large_group_participation_min,
        strong_p_max=strong_p_max,
    )

    ranked_consensus = rank_consensus_statements(
        vote_matrix=base_result.raw_vote_matrix,
        mod_out_statement_ids=kwargs.get("mod_out_statement_ids", []),
        fdr_rate=fdr_rate,
    )

    # Recompute GAC with effective agreement (agora improvement over Polis raw pa).
    n_groups = len(
        base_result.group_comment_stats.index.get_level_values("group_id").unique()
    )
    group_aware_consensus = compute_effective_agreement_gac(
        base_result.group_comment_stats,
        base_result.statements_df.index,
        n_groups,
    )

    return AgoraClusteringResult(
        raw_vote_matrix=base_result.raw_vote_matrix,
        filtered_vote_matrix=base_result.filtered_vote_matrix,
        reducer=base_result.reducer,
        clusterer=base_result.clusterer,
        group_comment_stats=base_result.group_comment_stats,
        statements_df=base_result.statements_df,
        participants_df=base_result.participants_df,
        participant_projections=base_result.participant_projections,
        statement_projections=base_result.statement_projections,
        group_aware_consensus=group_aware_consensus,
        ranked_repness=ranked_repness,
        ranked_consensus=ranked_consensus,
    )
