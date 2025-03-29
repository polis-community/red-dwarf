import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations import agora
from reddwarf.polis import PolisClient

# A helper to generate votes list.
def build_votes(data_fixture):
    _, data_path, *_ = data_fixture

    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    return client.data_loader.votes_data

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering_real_data_small(polis_convo_data):
    expected_cluster_sizes = [4, 10, 5]
    votes = build_votes(polis_convo_data)

    convo: agora.Conversation = {
        "id": "dummy",
        "votes": votes, # type:ignore
    }
    results = agora.run_clustering_v1(conversation=convo)

    assert len(expected_cluster_sizes) == len(results["clusters"])

    for cluster_id, expected_cluster_size in enumerate(expected_cluster_sizes):
        actual_cluster_size = len(results["clusters"][cluster_id]["participants"])
        assert actual_cluster_size == expected_cluster_size

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_run_clustering_real_data_medium(polis_convo_data):
    expected_cluster_sizes = [60, 39, 37, 37, 3]
    votes = build_votes(polis_convo_data)

    convo: agora.Conversation = {
        "id": "dummy",
        "votes": votes, # type:ignore
    }
    results = agora.run_clustering_v1(conversation=convo)

    assert len(expected_cluster_sizes) == len(results["clusters"])

    for cluster_id, expected_cluster_size in enumerate(expected_cluster_sizes):
        actual_cluster_size = len(results["clusters"][cluster_id]["participants"])
        assert actual_cluster_size == expected_cluster_size

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering_real_data_no_min_threshold_option(polis_convo_data):
    expected_cluster_sizes = [24, 7, 4]
    votes = build_votes(polis_convo_data)

    convo: agora.Conversation = {
        "id": "dummy",
        "votes": votes, # type:ignore
    }
    results = agora.run_clustering_v1(conversation=convo, options={"min_user_vote_threshold": 0})

    assert len(expected_cluster_sizes) == len(results["clusters"])

    for cluster_id, expected_cluster_size in enumerate(expected_cluster_sizes):
        actual_cluster_size = len(results["clusters"][cluster_id]["participants"])
        assert actual_cluster_size == expected_cluster_size

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_run_clustering_real_data_max_clusters_option(polis_convo_data):
    max_cluster_count = 3
    expected_cluster_count = max_cluster_count
    votes = build_votes(polis_convo_data)

    convo: agora.Conversation = {
        "id": "dummy",
        "votes": votes, # type:ignore
    }
    results = agora.run_clustering_v1(conversation=convo, options={"max_clusters": max_cluster_count})

    assert expected_cluster_count == len(results["clusters"])