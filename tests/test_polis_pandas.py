import json
from reddwarf.polis_pandas import PolisClient
from reddwarf import utils
import math

def test_user_vote_counts():
    # Load the expected data from the JSON file
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['user-vote-counts']
        # Convert keys from string to int. Not sure why API returns them as strings.
        expected_data = {int(k): v for k,v in expected_data.items()}

    # Instantiate the PolisClient and load raw data
    client = PolisClient()
    client.load_data(filepaths=['sample_data/below-100-ptpts/votes.json'])

    # Call the method and assert the result matches the expected data
    assert client.get_user_vote_counts() == expected_data

def test_meta_tids():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['meta-tids']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=['sample_data/below-100-ptpts/comments.json'])

    assert client.get_meta_tids() == sorted(expected_data)

def test_mod_in():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-in']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=['sample_data/below-100-ptpts/comments.json'])

    assert client.get_mod_in() == sorted(expected_data)

def test_mod_out():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-out']

    client = PolisClient(is_strict_moderation=True)
    client.load_data(filepaths=['sample_data/below-100-ptpts/comments.json'])

    assert sorted(client.get_mod_out()) == sorted(expected_data)

def test_last_vote_timestamp():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['lastVoteTimestamp']

    client = PolisClient()
    client.load_data(filepaths=['sample_data/below-100-ptpts/votes.json'])

    assert client.get_last_vote_timestamp() == expected_data

SHAPE_AXIS = { 'row': 0, 'column': 1 }

def test_participant_count():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n']

    client = PolisClient()
    client.load_data(filepaths=['sample_data/below-100-ptpts/votes.json'])
    client.get_matrix()

    assert client.participant_count == expected_data

def test_statement_count():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n-cmts']

    client = PolisClient()
    client.load_data(filepaths=['sample_data/below-100-ptpts/votes.json'])
    client.get_matrix()

    assert client.statement_count == expected_data

def test_impute_missing_values():
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        'sample_data/below-100-ptpts/votes.json',
        'sample_data/below-100-ptpts/comments.json',
    ])
    matrix_with_missing = client.get_matrix(is_filtered=True)
    matrix_without_missing = utils.impute_missing_votes(matrix_with_missing)

    assert matrix_with_missing.isnull().values.sum() > 0
    assert matrix_without_missing.isnull().values.sum() == 0
    assert matrix_with_missing.shape == matrix_without_missing.shape

def test_filtered_participants_grouped():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['in-conv']

    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        'sample_data/below-100-ptpts/votes.json',
        'sample_data/below-100-ptpts/comments.json',
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

def test_client_no_warnings_for_csv_urls(caplog):
    client = PolisClient(is_strict_moderation=True)
    client.load_data(report_id="r5hr48j8y8mpcffk7crmk", data_source="csv_export")
    assert len(caplog.records) == 0

def test_client_no_unexpected_warnings_for_api_urls(caplog):
    client = PolisClient()
    client.load_data(report_id="r5hr48j8y8mpcffk7crmk", data_source="api")
    # Filter out expected warning being thrown by urllib3 for Polis API.
    # See: https://github.com/urllib3/urllib3/blob/04662c9ae08c9f63fa254772d7618db65123a35e/src/urllib3/response.py#L693-L704
    # TODO: Figure out a way to hide these noisy WARNING's in caplog, for clarity on other test failures.
    expected_warnings = [r for r in caplog.records if "RFC 7230 sec 3.3.2" in r.message]
    assert len(caplog.records) - len(expected_warnings) == 0

    # Fail test if expected urllib warnings disappear, so we can update test and remove accomodation.
    assert len(expected_warnings) > 0

# def test_group_cluster_count():
#     with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['group-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/below-100-ptpts/votes.json')

#     assert len(client.get_group_clusters()) == len(expected_data)

# def test_pca_base_cluster_count():
#     with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['base-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/below-100-ptpts/votes.json')

#     assert len(client.base_clusters.get('x', [])) == len(expected_data['x'])
#     assert len(client.base_clusters.get('y', [])) == len(expected_data['y'])
