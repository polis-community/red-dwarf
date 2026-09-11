from collections.abc import Mapping, Sequence

from reddwarf.implementations import base


# This is to not break things.
# TODO: Adde deprecation warning.
def run_clustering(
    votes: Sequence[Mapping[str, object]],
    reducer: base.ReducerType = "pca",
    reducer_kwargs: dict[str, object] | None = None,
    clusterer: base.ClustererType = "kmeans",
    clusterer_kwargs: dict[str, object] | None = None,
    mod_out_statement_ids: list[int] | None = None,
    meta_statement_ids: list[int] | None = None,
    min_user_vote_threshold: int = 7,
    keep_participant_ids: list[int] | None = None,
    init_centers: list[list[float]] | None = None,
    max_group_count: int = 5,
    force_group_count: int | None = None,
    random_state: int | None = None,
    pick_max: int = 5,
    confidence: float = 0.9,
    consensus_mode: base.Literal["standard", "legacy"] = "legacy",
    candidate_group_counts: base.CandidateGroupCounts = None,
) -> base.TypedPolisPipelineResult:
    return run_pipeline(
        votes=votes,
        reducer=reducer,
        reducer_kwargs=reducer_kwargs,
        clusterer=clusterer,
        clusterer_kwargs=clusterer_kwargs,
        mod_out_statement_ids=mod_out_statement_ids,
        meta_statement_ids=meta_statement_ids,
        min_user_vote_threshold=min_user_vote_threshold,
        keep_participant_ids=keep_participant_ids,
        init_centers=init_centers,
        max_group_count=max_group_count,
        force_group_count=force_group_count,
        random_state=random_state,
        pick_max=pick_max,
        confidence=confidence,
        consensus_mode=consensus_mode,
        candidate_group_counts=candidate_group_counts,
    )


def run_pipeline(
    votes: Sequence[Mapping[str, object]],
    reducer: base.ReducerType = "pca",
    reducer_kwargs: dict[str, object] | None = None,
    clusterer: base.ClustererType = "kmeans",
    clusterer_kwargs: dict[str, object] | None = None,
    mod_out_statement_ids: list[int] | None = None,
    meta_statement_ids: list[int] | None = None,
    min_user_vote_threshold: int = 7,
    keep_participant_ids: list[int] | None = None,
    init_centers: list[list[float]] | None = None,
    max_group_count: int = 5,
    force_group_count: int | None = None,
    random_state: int | None = None,
    pick_max: int = 5,
    confidence: float = 0.9,
    consensus_mode: base.Literal["standard", "legacy"] = "legacy",
    candidate_group_counts: base.CandidateGroupCounts = None,
) -> base.TypedPolisPipelineResult:
    return base.run_pipeline(
        votes=list(votes),
        reducer=reducer,
        reducer_kwargs=reducer_kwargs or {},
        clusterer=clusterer,
        clusterer_kwargs=clusterer_kwargs or {},
        mod_out_statement_ids=mod_out_statement_ids or [],
        meta_statement_ids=meta_statement_ids or [],
        min_user_vote_threshold=min_user_vote_threshold,
        keep_participant_ids=keep_participant_ids or [],
        init_centers=init_centers,
        max_group_count=max_group_count,
        force_group_count=force_group_count,
        random_state=random_state,
        pick_max=pick_max,
        confidence=confidence,
        consensus_mode=consensus_mode,
        candidate_group_counts=candidate_group_counts,
    )
