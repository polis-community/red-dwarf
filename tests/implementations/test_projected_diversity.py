from unittest.mock import patch

import numpy as np
import pytest

from reddwarf.implementations import base


def _sparse_votes(*, polarized: bool) -> list[dict[str, int]]:
    return [
        {"participant_id": participant, "statement_id": statement, "vote": vote}
        for participant in range(6)
        for statement, vote in [
            (0, -1 if polarized and participant >= 3 else 1),
            (1 + participant % 3, 1),
        ]
    ]


@pytest.mark.parametrize("force_group_count", [None, 2])
def test_collapsed_projection_returns_insufficient_data(
    force_group_count: int | None,
) -> None:
    with patch.object(base, "run_clusterer", wraps=base.run_clusterer) as clusterer:
        result = base.run_pipeline(
            votes=_sparse_votes(polarized=False),
            min_user_vote_threshold=2,
            force_group_count=force_group_count,
            random_state=0,
        )

    assert isinstance(result, base.AnalysisInsufficientData)
    assert result.reason == base.InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS
    clusterer.assert_not_called()


def test_collapsed_candidate_batch_reuses_projection_without_clustering() -> None:
    with (
        patch.object(base, "run_reducer", wraps=base.run_reducer) as reducer,
        patch.object(base, "run_clusterer", wraps=base.run_clusterer) as clusterer,
    ):
        result = base.run_pipeline(
            votes=_sparse_votes(polarized=False),
            min_user_vote_threshold=2,
            candidate_group_counts=[2, 3],
            random_state=0,
        )

    assert isinstance(result, base.AnalysisSuccess)
    assert isinstance(result.result, base.KMeansCandidatesResult)
    assert [candidate.group_count for candidate in result.result.candidates] == [2, 3]
    for candidate in result.result.candidates:
        assert isinstance(candidate, base.KMeansCandidateInsufficientData)
        assert candidate.reason == base.InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS
    reducer.assert_called_once()
    clusterer.assert_not_called()


def test_mixed_candidates_keep_valid_groups_without_recomputing_pca() -> None:
    with (
        patch.object(base, "run_reducer", wraps=base.run_reducer) as reducer,
        patch.object(base, "run_clusterer", wraps=base.run_clusterer) as clusterer,
    ):
        result = base.run_pipeline(
            votes=_sparse_votes(polarized=True),
            min_user_vote_threshold=2,
            max_group_count=7,
            candidate_group_counts=[2, 4, 7],
            random_state=0,
        )

    assert isinstance(result, base.AnalysisSuccess)
    assert isinstance(result.result, base.KMeansCandidatesResult)
    valid, collapsed, too_few_samples = result.result.candidates
    assert isinstance(valid, base.KMeansCandidateSuccess)
    assert valid.result.participants_df["cluster_id"].nunique() == 2
    assert isinstance(collapsed, base.KMeansCandidateInsufficientData)
    assert collapsed.reason == base.InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS
    assert isinstance(too_few_samples, base.KMeansCandidateInsufficientData)
    assert too_few_samples.reason == (
        base.InsufficientDataReason.NOT_ENOUGH_SAMPLES_FOR_GROUP_COUNT
    )
    reducer.assert_called_once()
    clusterer.assert_called_once()


def test_reusable_projection_returns_typed_outcomes() -> None:
    projection = base.prepare_pca_projection(
        votes=_sparse_votes(polarized=True),
        min_user_vote_threshold=2,
        random_state=0,
    )
    valid = base.run_kmeans_on_pca_projection(projection=projection, force_group_count=2)
    invalid = base.run_kmeans_on_pca_projection(projection=projection, force_group_count=4)

    assert isinstance(valid, base.AnalysisSuccess)
    assert valid.result.participants_df["cluster_id"].nunique() == 2
    assert isinstance(invalid, base.AnalysisInsufficientData)
    assert invalid.reason == base.InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS


def test_forced_pipeline_rejects_too_many_projected_groups() -> None:
    result = base.run_pipeline(
        votes=_sparse_votes(polarized=True),
        min_user_vote_threshold=2,
        force_group_count=4,
        random_state=0,
    )
    assert isinstance(result, base.AnalysisInsufficientData)
    assert result.reason == base.InsufficientDataReason.NOT_ENOUGH_UNIQUE_POINTS


def test_automatic_group_search_is_bounded_by_projected_diversity() -> None:
    with patch.object(base, "run_clusterer", wraps=base.run_clusterer) as clusterer:
        result = base.run_pipeline(
            votes=_sparse_votes(polarized=True),
            min_user_vote_threshold=2,
            max_group_count=5,
            random_state=0,
        )

    assert isinstance(result, base.AnalysisSuccess)
    assert isinstance(result.result, base.PolisClusteringResult)
    coordinates = result.result.participants_df.loc[:, ["x", "y"]].values
    clusterer.assert_called_once()
    assert clusterer.call_args.kwargs["max_group_count"] == len(
        np.unique(coordinates, axis=0)
    )


def test_unrelated_clustering_errors_propagate() -> None:
    with (
        patch.object(base, "run_clusterer", side_effect=ValueError("unrelated failure")),
        pytest.raises(ValueError, match="unrelated failure"),
    ):
        base.run_pipeline(
            votes=_sparse_votes(polarized=True),
            min_user_vote_threshold=2,
            candidate_group_counts=[2, 4],
            random_state=0,
        )
