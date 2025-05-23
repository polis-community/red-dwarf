import numpy as np
import pandas as pd
import pytest
from tests.fixtures import polis_convo_data
from reddwarf.utils import stats

from reddwarf.utils import stats, polismath, matrix
from reddwarf.data_loader import Loader

def test_importance_metric_no_votes():
    expected_importance = [ 1/4,   2/4,   1,     2,      4   ]
    comment_extremity =   [(1-1), (2-1), (4-1), (8-1), (16-1)]
    # extremity values    [ 0,     1,     3,     7,     15   ]

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=0,
        extremity=comment_extremity,
    )

    assert expected_importance == calculated_importance.tolist()

def test_importance_metric_limits_no_extremity_all_agree():
    comment_extremity = 0
    expected_importance = 1

    calculated_importance = stats.importance_metric(
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_all_disagree():
    comment_extremity = 0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_split_full_engagement():
    comment_extremity = 0
    expected_importance = 1/4

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=5000,
        n_disagree=5000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_all_pass():
    comment_extremity = 0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_agree():
    comment_extremity = 4.0
    expected_importance = comment_extremity+1

    calculated_importance = stats.importance_metric(
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_disagree():
    comment_extremity = 4.0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_pass():
    comment_extremity = 4.0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

# TODO: Use this test to more fully show trends and effects.
@pytest.mark.skip()
def test_importance_metric_array():
    expected_importances = [0, 7**2]

    calculated_priority = stats.importance_metric(
        n_agree=   [    0,     0],
        n_disagree=[10000, 10000],
        n_total=   [10000, 10000],
        extremity= [  4.0,   4.0],
    )
    assert calculated_priority == pytest.approx(expected_importances, abs=0.001)

def test_importance_metric_smaller_full_agree_pseudo_count():
    # Should approach 1 at higher volume of votes
    pseudo_counts =        [1,     10]
    # Approaches slower with higher pseudo-count.
    expected_importances = [0.9804, 0.84027778]

    calculated_priority = stats.importance_metric(
        n_agree=   [100, 100],
        n_disagree=[  0,   0],
        n_total=   [100, 100],
        extremity= [  0,   0],
        pseudo_count=pseudo_counts,
    )
    assert calculated_priority == pytest.approx(expected_importances, abs=0.001)

def test_priority_metric_no_votes():
    prio = lambda n: (81/16)*(4**n)
    # expected_values = [ 5.0625,  20.25,   81,      324,     1296   ]
    expected_priority = [ prio(0), prio(1), prio(2), prio(3), prio(4)]
    comment_extremity = [ 0,       1,       (4-1),   (8-1),   (16-1) ]
    # extremity values  [ 0,       1,        3,       7,       15    ]

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=0,
        extremity=comment_extremity,
    )

    assert expected_priority == calculated_priority.tolist()

# TODO: Investigate why "medium-with-meta" and "medium-no-meta" don't pass.
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"], indirect=True)
def test_priority_metric_real_data(polis_convo_data):
    fixture = polis_convo_data
    votes_base = fixture.math_data["votes-base"]
    # Get index and statement_id because polismath lists (like pca sub-keys) are
    # indexed, and polismath objects (like priorities) are keyed to statement_id
    for idx, (statement_id, votes) in enumerate(votes_base.items()):
        expected_priority = fixture.math_data["comment-priorities"][statement_id]

        is_meta = int(statement_id) in fixture.math_data["meta-tids"]
        n_agree = np.asarray(votes["A"]).sum()
        n_disagree = np.asarray(votes["D"]).sum()
        n_total = np.asarray(votes["S"]).sum()
        comment_extremity = fixture.math_data["pca"]["comment-extremity"][idx]

        calculated_priority = stats.priority_metric(
            is_meta=is_meta,
            n_agree=n_agree,
            n_disagree=n_disagree,
            n_total=n_total,
            extremity=comment_extremity,
        )
        assert expected_priority == pytest.approx(calculated_priority)

def test_priority_metric_for_meta_default():
    meta_priority_default = 7
    expected_priority = meta_priority_default**2

    calculated_priority = stats.priority_metric(
        is_meta=True,
        n_agree=10,
        n_disagree=0,
        n_total=25,
        extremity=0,
    )

    assert calculated_priority == expected_priority

def test_priority_metric_for_meta_override():
    meta_priority_override = 10
    expected_priority = meta_priority_override**2

    calculated_priority = stats.priority_metric(
        is_meta=True,
        n_agree=10,
        n_disagree=0,
        n_total=25,
        extremity=1.0,
        meta_priority=meta_priority_override,
    )
    assert calculated_priority == expected_priority

def test_priority_metric_limits_no_extremity_all_passing():
    comment_extremity = 0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority)

def test_priority_metric_limits_no_extremity_all_disagree():
    comment_extremity = 0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_no_extremity_all_agree():
    comment_extremity = 0
    expected_priority = 1

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_no_extremity_split_full_engagement():
    comment_extremity = 0
    expected_priority = 1/4

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=5000,
        n_disagree=5000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_high_extremity_all_passed():
    comment_extremity = 4.0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority)

def test_priority_metric_limits_high_extremity_all_agree():
    comment_extremity = 4.0
    expected_priority = (comment_extremity+1)**2

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.01)

def test_priority_metric_limits_high_extremity_all_disagree():
    comment_extremity = 4.0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

# TODO: Use this test to more fully show trends and effects.
def test_priority_metric_array():
    expected_priorities = [0, 7**2]

    calculated_priority = stats.priority_metric(
        is_meta=   [False,  True],
        n_agree=   [    0,     0],
        n_disagree=[10000, 10000],
        n_total=   [10000, 10000],
        extremity= [  4.0,   4.0],
    )
    assert calculated_priority == pytest.approx(expected_priorities, abs=0.001)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"], indirect=True)
def test_group_aware_consensus_real_data(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[
        f'{fixture.data_dir}/votes.json',
        f'{fixture.data_dir}/comments.json',
        f'{fixture.data_dir}/conversation.json',
    ])
    VOTES = loader.votes_data
    vote_matrix = matrix.generate_raw_matrix(votes=VOTES)

    # TODO: Why do moderated out statements not plug into comment stats? BUG?
    # STATEMENTS = loader.comments_data
    # _, _, mod_out, _ = stmnts.process_statements(statement_data=STATEMENTS)
    # vote_matrix = matrix.simple_filter_matrix(
    #     vote_matrix=vote_matrix,
    #     mod_out_statement_ids=mod_out,
    # )

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_clustered_participant_ids, cluster_labels = polismath.extract_data_from_polismath(fixture.math_data)
    vote_matrix = vote_matrix.loc[all_clustered_participant_ids, :]

    # Generate stats all groups and all statements.
    _, gac_df = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )

    calculated_gac = {
        str(pid): float(row.iloc[0])
        for pid, row in gac_df.iterrows()
    }

    assert calculated_gac == pytest.approx(fixture.math_data["group-aware-consensus"])

def test_format_comment_stats_repful_agree():
    statement = pd.Series({
        "statement_id": 1,
        "ns": 100,
        "na": 60, "pa": 0.6, "pat": 2.0,
        "nd": 40, "pd": 0.4, "pdt": 1.5,
        "ra": 0.8, "rat": 3.0,
        "rd": 0.5, "rdt": 2.0,
    })

    result = stats.format_comment_stats(statement)
    assert result == {
        "tid": 1,
        "n-success": 60,
        "n-trials": 100,
        "p-success": 0.6,
        "p-test": 2.0,
        "repness": 0.8,
        "repness-test": 3.0,
        "repful-for": "agree",
    }

def test_format_comment_stats_repful_disagree():
    statement = pd.Series({
        "statement_id": 2,
        "ns": 100,
        "na": 45, "pa": 0.45, "pat": 1.7,
        "nd": 55, "pd": 0.55, "pdt": 2.0,
        "ra": 0.5, "rat": 1.8,
        "rd": 0.7, "rdt": 2.5,
    })

    result = stats.format_comment_stats(statement)
    assert result == {
        "tid": 2,
        "n-success": 55,
        "n-trials": 100,
        "p-success": 0.55,
        "p-test": 2.0,
        "repness": 0.7,
        "repness-test": 2.5,
        "repful-for": "disagree",
    }

def test_format_comment_stats_consensus_agree():
    statement = pd.Series({
        "statement_id": 3,
        "ns": 100,
        "na": 70, "pa": 0.7, "pat": 2.2,
        "nd": 30, "pd": 0.3, "pdt": 1.0,
    })

    result = stats.format_comment_stats(statement)
    assert result == {
        "tid": 3,
        "n-success": 70,
        "n-trials": 100,
        "p-success": 0.7,
        "p-test": 2.2,
        "cons-for": "agree",
    }

def test_format_comment_stats_consensus_disagree():
    statement = pd.Series({
        "statement_id": 4,
        "ns": 100,
        "na": 40, "pa": 0.4, "pat": 1.2,
        "nd": 60, "pd": 0.6, "pdt": 2.3,
    })

    result = stats.format_comment_stats(statement)
    assert result == {
        "tid": 4,
        "n-success": 60,
        "n-trials": 100,
        "p-success": 0.6,
        "p-test": 2.3,
        "cons-for": "disagree",
    }