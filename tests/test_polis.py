import json
from reddwarf.polis import PolisClient

def test_user_vote_counts():
    # Load the expected data from the JSON file
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['user-vote-counts']
        # Convert keys from string to int. Not sure why API returns them as strings.
        expected_data = {int(k): v for k,v in expected_data.items()}

    # Instantiate the PolisClient and load raw data
    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/votes.json')

    # Call the method and assert the result matches the expected data
    assert client.get_user_vote_counts() == expected_data

def test_meta_tids():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['meta-tids']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/comments.json')

    assert client.get_meta_tids() == expected_data

def test_mod_in():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-in']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/comments.json')

    assert client.get_mod_in() == sorted(expected_data)

def test_mod_out():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-out']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/comments.json')

    assert sorted(client.get_mod_out()) == sorted(expected_data)

def test_last_vote_timestamp():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['lastVoteTimestamp']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/votes.json')

    assert client.get_last_vote_timestamp() == expected_data

SHAPE_AXIS = { 'row': 0, 'column': 1 }

def test_participant_count():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/votes.json')
    client.get_matrix()

    assert client.participant_count == expected_data

def test_statement_count():
    with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n-cmts']

    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/votes.json')
    client.get_matrix()

    assert client.statement_count == expected_data

def test_impute_missing_values():
    client = PolisClient()
    client.load_data('sample_data/above-100-ptpts/votes.json')
    matrix_with_missing = client.get_matrix()
    client.impute_missing_votes()
    matrix_without_missing = client.get_matrix()

    assert matrix_with_missing.null_count().sum_horizontal().item() > 0
    assert matrix_without_missing.null_count().sum_horizontal().item() == 0
    assert matrix_with_missing.shape == matrix_without_missing.shape

# def test_group_cluster_count():
#     with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['group-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/above-100-ptpts/votes.json')

#     assert len(client.get_group_clusters()) == len(expected_data)

# def test_pca_base_cluster_count():
#     with open('sample_data/above-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['base-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/above-100-ptpts/votes.json')

#     assert len(client.base_clusters.get('x', [])) == len(expected_data['x'])
#     assert len(client.base_clusters.get('y', [])) == len(expected_data['y'])
