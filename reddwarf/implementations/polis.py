from typing import Optional
from pandas import DataFrame
from sklearn.decomposition import PCA
from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.utils.matrix import generate_raw_matrix, simple_filter_matrix, get_clusterable_participant_ids
from reddwarf.utils.pca import run_pca
from reddwarf.utils.clustering import find_optimal_k
from dataclasses import dataclass
import pandas as pd

from reddwarf.utils.stats import calculate_comment_statistics_dataframes, populate_priority_calculations_into_statements_df

@dataclass
class PolisClusteringResult:
    """
    Attributes:
        raw_vote_matrix (DataFrame): Raw sparse vote matrix before any processing.
        filtered_vote_matrix (DataFrame): Raw sparse vote matrix with moderated statements zero'd out.
        pca (PCA): PCA model fitted to vote matrix, including `mean_`, `expected_variance_` (eigenvalues) and `components_` (eigenvectors).
        projected_participants (DataFrame): Dataframe of projected participants, with columns "x", "y", "cluster_id"
        projected_statements (DataFrame): Dataframe of projected statements, with columns "x", "y".
        kmeans (PolisKMeans): Scikit-Learn KMeans object for selected group count, including `labels_` and `cluster_centers_`. See `PolisKMeans`.
        group_aware_consensus (DataFrame): Group-aware consensus scores for each statement.
        group_comment_stats (DataFrame): A multi-index dataframes for each statement, indexed by group ID and statement.
        statements_df (DataFrame): A dataframe with all intermediary and final statement data/calculations/metadata.
        participants_df (DataFrame): A dataframe with all intermediary and final participant data/calculations/metadata.
    """
    raw_vote_matrix: DataFrame
    filtered_vote_matrix: DataFrame
    pca: PCA
    projected_participants: DataFrame
    projected_statements: DataFrame
    kmeans: PolisKMeans | None
    group_aware_consensus: DataFrame
    group_comment_stats: DataFrame
    statements_df: DataFrame
    participants_df: DataFrame

def run_clustering(
    votes: list[dict],
    mod_out_statement_ids: list[int] = [],
    meta_statement_ids: list[int] = [],
    min_user_vote_threshold: int = 7,
    keep_participant_ids: list[int] = [],
    init_centers: Optional[list[list[float]]] = None,
    max_group_count: int = 5,
    force_group_count: Optional[int] = None,
    random_state: Optional[int] = None,
    previous_result: Optional[PolisClusteringResult] = None,
) -> PolisClusteringResult:
    """
    An essentially feature-complete implementation of the Polis clustering algorithm.

    Still missing:
        - base-cluster calculations (so can't match output of conversations larger than 100 participants),
        - k-smoothing, which holds back k-value (group count) until re-calculated 3 consecutive times,
        - some advanced participant filtering that involves past state (you can use keep_participant_ids to mimic manually).

    Args:
        votes (list[dict]): Raw list of vote dicts, with keys for "participant_id", "statement_id", "vote" and "modified"
        mod_out_statement_ids (list[int]): List of statement IDs to moderate/zero out
        meta_statement_ids (list[int]): List of meta statement IDs
        min_user_vote_threshold (int): Minimum number of votes a participant must make to be included in clustering
        keep_participant_ids (list[int]): List of participant IDs to keep in clustering algorithm, regardless of normal filters.
        max_group_count (): Max number of group (k-values) to test using k-means and silhouette scores
        init_centers (list[list[float]]): Initial guesses of [x,y] coordinates for k-means (Length of list must match max_group_count)
        force_group_count (int): Instead of using silhouette scores, force a specific number of groups (k value)
        random_state (int): If set, will force determinism during k-means clustering
        previous_result (PolisClusteringResult): The result of a previous run of this function, to seed args.

    Returns:
        PolisClusteringResult: A dataclass containing clustering results, including intermediate calculations.
    """
    if previous_result:
        prev_kmeans = previous_result.kmeans
        if prev_kmeans:
            init_centers = prev_kmeans.cluster_centers_
            # TODO: Implement some variant of k-smoothing to stabilize this instead of locking it.
            prev_k = len(prev_kmeans.cluster_centers_)
            force_group_count = prev_k
        keep_participant_ids = list(previous_result.participants_df.query('to_cluster').index)

    raw_vote_matrix = generate_raw_matrix(votes=votes)

    filtered_vote_matrix = simple_filter_matrix(
        vote_matrix=raw_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    # Run PCA and generate participant/statement projections.
    # DataFrames each have "x" and "y" columns.
    participants_df, statements_df, pca = run_pca(vote_matrix=filtered_vote_matrix)

    participant_ids_to_cluster = get_clusterable_participant_ids(raw_vote_matrix, vote_threshold=min_user_vote_threshold)
    if keep_participant_ids:
        # TODO: Make this an intersection, in case there are members of
        # keep_participant_ids list that aren't represented in vote_matrix.
        participant_ids_to_cluster = sorted(list(set(participant_ids_to_cluster + keep_participant_ids)))

    if force_group_count:
        k_bounds = [force_group_count, force_group_count]
    else:
        k_bounds = [2, max_group_count]

    _, _, kmeans = find_optimal_k(
        projected_data=participants_df.loc[participant_ids_to_cluster, :],
        k_bounds=k_bounds,
        # Force polis strategy of initiating cluster centers. See: PolisKMeans.
        init="polis",
        init_centers=init_centers,
        random_state=random_state,
    )
    label_series = pd.Series(
        kmeans.labels_ if kmeans else None,
        index=participant_ids_to_cluster,
        dtype="Int64", # Allows nullable/NaN values.
    )
    participants_df["to_cluster"] = participants_df.index.isin(participant_ids_to_cluster)
    participants_df["cluster_id"] = label_series

    grouped_stats_df, gac_df = calculate_comment_statistics_dataframes(
        vote_matrix=raw_vote_matrix.loc[participant_ids_to_cluster, :],
        cluster_labels=kmeans.labels_,
    )

    def get_with_default(lst, idx, default=None):
        try:
            return lst[idx]
        except IndexError:
            return default

    statements_df["to_zero"] = statements_df.index.isin(mod_out_statement_ids)
    statements_df["is_meta"] = statements_df.index.isin(meta_statement_ids)
    statements_df["mean"] = pca.mean_
    statements_df["pc1"] = get_with_default(pca.components_, 0)
    statements_df["pc2"] = get_with_default(pca.components_, 1)
    statements_df["pc3"] = get_with_default(pca.components_, 2)
    statements_df = pd.concat([statements_df, gac_df], axis=1)
    statements_df = populate_priority_calculations_into_statements_df(
        statements_df=statements_df,
        vote_matrix=raw_vote_matrix.loc[participant_ids_to_cluster, :],
    )

    return PolisClusteringResult(
        raw_vote_matrix=raw_vote_matrix,
        filtered_vote_matrix=filtered_vote_matrix,
        pca=pca,
        projected_participants=participants_df.loc[participant_ids_to_cluster, ["x", "y", "cluster_id"]], # deprecate?
        projected_statements=statements_df.loc[:, ["x", "y"]], # deprecate?
        kmeans=kmeans,
        group_aware_consensus=gac_df, # deprecate?
        group_comment_stats=grouped_stats_df,
        statements_df=statements_df,
        participants_df=participants_df,
    )