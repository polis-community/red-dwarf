from reddwarf import utils
from reddwarf.exceptions import RedDwarfError

import pytest
from pandas.testing import assert_frame_equal
import pandas as pd
from collections import namedtuple
from random import shuffle
from numpy import nan

statement_count = lambda vote_matrix: vote_matrix.shape[1]
participant_count = lambda vote_matrix: vote_matrix.shape[0]

@pytest.fixture()
def make_votes():
    DatedVote = namedtuple("DatedVote", ["participant_id", "statement_id", "vote", "modified"])

    def make(
        id_type: str = 'int',
        include_timestamp: bool = True,
    ):
        votes_str = [
            DatedVote("p1", "s1",  0, 1730000000),
            DatedVote("p1", "s2",  1, 1730000000),
            DatedVote("p3", "s1", -1, 1730000000),
            DatedVote("p2", "s3",  1, 1735000000),
            DatedVote("p2", "s1", -1, 1735700000),
            DatedVote("p2", "s2",  1, 1735700000),
        ]

        votes_int = [
            DatedVote(0, 0,  0, 1730000000),
            DatedVote(0, 1,  1, 1730000000),
            DatedVote(2, 0, -1, 1730000000),
            DatedVote(1, 2,  1, 1735000000),
            DatedVote(1, 0, -1, 1735700000),
            DatedVote(1, 1,  1, 1735700000),
        ]

        votes_int_offset = [
            DatedVote(1, 2,  0, 1730000000),
            DatedVote(1, 5,  1, 1730000000),
            DatedVote(6, 2, -1, 1730000000),
            DatedVote(4, 7,  1, 1735000000),
            DatedVote(4, 2, -1, 1735700000),
            DatedVote(4, 5,  1, 1735700000),
        ]

        votes = {
            "str": votes_str,
            "int": votes_int,
            "int_offset": votes_int_offset,
        }.get(id_type, "int")

        # Convert to dict
        votes = [v._asdict() for v in votes]

        if include_timestamp == False:
            votes = [
                { k: v for k, v in d.items() if k != 'modified' }
                for d in votes
            ]

        return votes

    return make

@pytest.fixture()
def make_vote_matrix():

    def make(id_type: str = 'int'):
        ids_str = (
            ["p1", "p2", "p3"], # participant IDs, row index
            ["s1", "s2", "s3"], # statement IDs, column name
        )
        ids_int = (
            [0, 1, 2],
            [0, 1, 2],
        )
        ids_int_offset = (
            [1, 4, 6],
            [2, 5, 7],
        )

        ids = {
            "str": ids_str,
            "int": ids_int,
            "int_offset": ids_int_offset,
        }.get(id_type, "int")

        # Convert to dict
        participant_ids, statement_ids = ids
        vote_matrix = pd.DataFrame(
            [
                [ 0,    1, None],
                [-1,    1,    1],
                [-1, None, None],
            ],
            index=participant_ids, # participant_ids
            columns=statement_ids, # statement_ids
            dtype=float,
        )
        # Reproduce property of pivot columns.
        vote_matrix.index.name = "participant_id"
        vote_matrix.columns.name = "statement_id"
        return vote_matrix

    return make


def test_generate_raw_matrix_simple_string_identifiers(make_votes, make_vote_matrix):
    simple_votes = make_votes(id_type='str', include_timestamp=False)
    simple_vote_matrix = make_vote_matrix(id_type='str')
    expected = simple_vote_matrix
    vote_matrix = utils.generate_raw_matrix(votes=simple_votes)
    assert_frame_equal(vote_matrix, expected)

def test_generate_raw_matrix_simple_int_identifiers(make_votes, make_vote_matrix):
    simple_votes = make_votes(id_type='int', include_timestamp=False)
    simple_vote_matrix = make_vote_matrix(id_type='int')

    expected = simple_vote_matrix
    vote_matrix = utils.generate_raw_matrix(votes=simple_votes)
    assert_frame_equal(vote_matrix, expected)

@pytest.mark.skip
def test_filter_votes():
    raise

# TODO: Convert these to unit tests of filter_votes()?
def test_generate_raw_matrix_cutoff_from_timestamp(make_votes):
    simple_timestamped_votes = make_votes(include_timestamp=True)
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    last_vote_timestamp = 1735000000
    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=last_vote_timestamp)

    vote_before = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_index_start(make_votes):
    simple_timestamped_votes = make_votes(include_timestamp=True)
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=4)

    vote_before = simple_timestamped_votes[3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[4]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_index_end(make_votes):
    simple_timestamped_votes = make_votes(include_timestamp=True)
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=-2)

    third_last_vote = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[third_last_vote["participant_id"], third_last_vote["statement_id"]])

    second_last_vote = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[second_last_vote["participant_id"], second_last_vote["statement_id"]])

def test_generate_raw_matrix_cutoff_from_timestamp_missing_error(make_votes):
    simple_timestamped_votes = make_votes(include_timestamp=False)
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    last_vote_timestamp = 1735000000
    with pytest.raises(RedDwarfError):
        utils.generate_raw_matrix(votes=shuffled_votes, cutoff=last_vote_timestamp)

def test_generate_raw_matrix_cutoff_from_index_missing_error(make_votes):
    simple_timestamped_votes = make_votes(include_timestamp=False)
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    with pytest.raises(RedDwarfError):
        utils.generate_raw_matrix(votes=shuffled_votes, cutoff=-4)

def test_get_unvoted_statement_ids(make_vote_matrix):
    simple_vote_matrix = make_vote_matrix()
    # Null out existing statement column.
    simple_vote_matrix[1] = nan
    # Null out new statement column.
    simple_vote_matrix[3] = None
    unused_statement_ids = simple_vote_matrix.pipe(utils.get_unvoted_statement_ids)
    assert unused_statement_ids == [1, 3]

def test_filter_matrix_no_filtering(make_vote_matrix):
    initial_matrix = make_vote_matrix()
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=0,
    )

    # No filtering
    assert participant_count(initial_matrix) == participant_count(filtered_matrix)
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_min_user_vote(make_vote_matrix):
    initial_matrix = make_vote_matrix()
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=2,
    )

    below_threshold_participant_id = 2
    assert below_threshold_participant_id in initial_matrix.index
    assert below_threshold_participant_id not in filtered_matrix.index

    # No statements filtered
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_min_user_vote_bypass(make_vote_matrix):
    initial_matrix = make_vote_matrix()
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=2,
        keep_participant_ids=[2],
    )

    below_threshold_participant_id = 2
    assert below_threshold_participant_id in initial_matrix.index
    assert below_threshold_participant_id in filtered_matrix.index

    # No statements filtered
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_mod_out_statement(make_vote_matrix):
    initial_matrix = make_vote_matrix()
    all_statement_ids_but_one = initial_matrix.columns[:-1]
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids_but_one,
        min_user_vote_threshold=0,
    )

    assert participant_count(initial_matrix) == participant_count(filtered_matrix)

    assert statement_count(initial_matrix) == 3
    assert statement_count(filtered_matrix) == 2

def test_filter_matrix_mod_out_statement_with_zero(make_vote_matrix):
    initial_matrix = make_vote_matrix()
    # Set statement 1 to have no votes.
    initial_matrix[1] = None
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=0,
        unvoted_filter_type="zero",
    )
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

    assert (filtered_matrix[1] == 0).all()

def test_impute_missing_votes():
    # Manually calculated
    expected_matrix = pd.DataFrame(
        [
            [ 0,    1,    0,  1/3],
            [-1,    0,    0,   -1],
            [-1,  0.5,    0,    1],
            [-1,  0.5,    0,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )

    initial_matrix = pd.DataFrame(
        [
            [ 0,    1, None, None],
            [-1,    0, None,   -1],
            [-1, None, None,    1],
            [-1, None,    0,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )
    imputed_matrix = utils.impute_missing_votes(vote_matrix=initial_matrix)
    assert_frame_equal(imputed_matrix, expected_matrix)

def test_impute_missing_votes_no_vote_statement_error():
    initial_matrix = pd.DataFrame(
        [
            [ 0,    1, None, None],
            [-1,    0, None,   -1],
            [-1, None, None,    1],
            [-1, None, None,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )
    with pytest.raises(RedDwarfError):
        utils.impute_missing_votes(vote_matrix=initial_matrix)

@pytest.mark.skip
def test_run_kmeans():
    raise

@pytest.mark.skip
def test_find_optimal_k():
    raise
