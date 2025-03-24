from reddwarf.polis import PolisClient
from reddwarf import utils
import math
import pytest

from tests.fixtures import polis_convo_data

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_user_vote_counts(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['user-vote-counts']
    expected_data = {int(k): v for k,v in expected_data.items()}

    # Instantiate the PolisClient and load raw data
    client = PolisClient()
    client.load_data(filepaths=[f'{path}/votes.json'])

    # Call the method and assert the result matches the expected data
    assert client.get_user_vote_counts() == expected_data

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_meta_tids(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['meta-tids']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=[f'{path}/comments.json'])

    assert client.get_meta_tids() == sorted(expected_data)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_mod_in(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['mod-in']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=[f'{path}/comments.json'])

    assert client.get_mod_in() == sorted(expected_data)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_mod_out(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['mod-out']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=[f'{path}/comments.json'])

    assert sorted(client.get_mod_out()) == sorted(expected_data)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_last_vote_timestamp(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['lastVoteTimestamp']

    client = PolisClient()
    client.load_data(filepaths=[f'{path}/votes.json'])

    assert client.get_last_vote_timestamp() == expected_data

SHAPE_AXIS = { 'row': 0, 'column': 1 }

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_participant_count(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['n']

    client = PolisClient()
    client.load_data(filepaths=[f'{path}/votes.json'])
    client.get_matrix()

    assert client.participant_count == expected_data

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_statement_count(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['n-cmts']

    client = PolisClient()
    client.load_data(filepaths=[f'{path}/votes.json'])
    client.get_matrix()

    assert client.statement_count == expected_data

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_impute_missing_values(polis_convo_data):
    _, path, _ = polis_convo_data
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])
    matrix_with_missing = client.get_matrix(is_filtered=True)
    matrix_without_missing = utils.impute_missing_votes(matrix_with_missing)

    assert matrix_with_missing.isnull().values.sum() > 0
    assert matrix_without_missing.isnull().values.sum() == 0
    assert matrix_with_missing.shape == matrix_without_missing.shape

# This test can't be paramtrized without changes.
@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_filtered_participants_grouped(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['in-conv']

    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])

    unaligned_matrix = client.get_matrix(is_filtered=True)
    assert sorted(unaligned_matrix.index.to_list()) != sorted(expected_data)

    client.matrix = None
    client.keep_participant_ids = [ 5, 10, 11, 14 ]
    aligned_matrix = client.get_matrix(is_filtered=True)
    assert sorted(aligned_matrix.index.to_list()) == sorted(expected_data)

def test_infer_moderation_type_from_api():
    client = PolisClient()
    assert client.is_strict_moderation is None
    client.load_data(conversation_id="9knpdktubt")
    assert client.is_strict_moderation is not None

def test_load_data_from_report_id():
    client = PolisClient()
    client.load_data(report_id="r5hr48j8y8mpcffk7crmk")

def test_matrix_cutoff_timestamp():
    client = PolisClient()
    client.load_data(conversation_id="9knpdktubt")

    full_matrix = client.get_matrix()

    cutoff_timestamp = 1528749254597
    client.matrix = None
    cutoff_matrix = client.get_matrix(cutoff=cutoff_timestamp)

    is_past_cutoff = lambda x: x["modified"] > cutoff_timestamp
    votes_after = list(filter(is_past_cutoff, client.votes))
    vote_after = votes_after[0]

    assert math.isnan(cutoff_matrix.loc[vote_after["participant_id"], vote_after["statement_id"]])
    assert not math.isnan(full_matrix.loc[vote_after["participant_id"], vote_after["statement_id"]])

class Test_Client:
    # Was getting a wall of warnings due to requests-cache bug.
    # Checking this to ensure it doesn't return and confuse users.
    # Fixed here: https://github.com/requests-cache/requests-cache/pull/1068
    def test_client_no_warnings_for_csv_urls(self, caplog):
        client = PolisClient(is_strict_moderation=True)
        client.load_data(report_id="r5hr48j8y8mpcffk7crmk", data_source="csv_export")
        assert len(caplog.records) == 0

    def test_client_no_warnings_for_api_urls(self, caplog):
        client = PolisClient()
        client.load_data(report_id="r5hr48j8y8mpcffk7crmk", data_source="api")
        assert len(caplog.records) == 0

@pytest.mark.skip
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_group_cluster_count(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['group-clusters']

    client = PolisClient()
    client.load_data(f'{path}/votes.json')

    assert len(client.get_group_clusters()) == len(expected_data)

@pytest.mark.skip
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_pca_base_cluster_count(polis_convo_data):
    data, path, _ = polis_convo_data
    expected_data = data['base-clusters']

    client = PolisClient()
    client.load_data(f'{path}/votes.json')

    assert len(client.base_clusters.get('x', [])) == len(expected_data['x'])
    assert len(client.base_clusters.get('y', [])) == len(expected_data['y'])
