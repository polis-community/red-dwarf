import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.polis import PolisClient
from reddwarf.utils.statements import process_statements
from reddwarf.utils.polismath import extract_data_from_polismath
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


def transform_base_clusters_to_participant_coords(base_clusters):
    """
    Transform base clusters data into a list of dictionaries with participant_id and xy coordinates.

    Args:
        base_clusters (dict): A dictionary containing base clusters data with 'id', 'members', 'x', and 'y' keys.

    Returns:
        list: A list of dictionaries, each containing a participant_id and their xy coordinates.
    """
    # For now, ensure failure if a base-cluster has more than one member, as the test assumes that.
    def get_only_member_or_raise(members):
        if len(members) != 1:
            raise Exception("A base-cluster has more than one member when it cannot")

        return members[0]

    return [
        {
            "participant_id": get_only_member_or_raise(members),
            "xy": [x, y]
        }
        for members, x, y in zip(
            base_clusters['members'],
            base_clusters['x'],
            base_clusters['y']
        )
    ]

def pad_to_size(lst, size):
    return list(lst) + [[0., 0.]]*(size - len(lst))

# This test will only match polismath for sub-100 participant convos.
# For testing our code agaisnt real data with against larger conversations,
# we'll need to implement base clustering.
@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering_real_data(polis_convo_data):
    math_data, data_path, _ = polis_convo_data

    # We hardcode this because Polis has some bespoke rules that keep these IDs in for clustering.
    # TODO: Try to determine why each pid is kept. Can maybe determine by incrementing through vote history.
    #  5 -> 1 vote @ statement #26 (participant #2's)
    # 10 -> 2 vote @ statements #21 (participant #1's) & #29 (their own, moderated in).
    # 11 -> 1 vote @ statement #29 (participant #10's)
    # 14 -> 1 vote @ statement #27 (participant #6's)
    keep_participant_ids = [ 5, 10, 11, 14 ]
    # We force kmeans to find a specific value because Polis does this for something they call k-smoothing.
    # TODO: Get k-smoothing working, and simulate how k is held back.
    force_group_count = len(math_data["group-clusters"])

    # Transpose base cluster coords into participant_ids
    expected_projected_ptpts = transform_base_clusters_to_participant_coords(math_data["base-clusters"])

    max_group_count = 5
    init_centers = [group["center"] for group in math_data["group-clusters"]]
    init_centers = pad_to_size(init_centers, max_group_count)

    client = PolisClient()
    client.load_data(filepaths=[
        f"{data_path}/votes.json",
        # Loading these helps generate mod_out_statement_ids
        f"{data_path}/comments.json",
        f"{data_path}/conversation.json",
    ])

    _, _, mod_out_statement_ids, _ = process_statements(statement_data=client.data_loader.comments_data)

    projected_ptpts, comps, _, center, _ = run_clustering(
        votes=client.data_loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        keep_participant_ids=keep_participant_ids,
        max_group_count=max_group_count,
        init_centers=init_centers,
        force_group_count=force_group_count,
    )

    assert comps[0] == pytest.approx(math_data["pca"]["comps"][0])
    assert comps[1] == pytest.approx(math_data["pca"]["comps"][1])
    assert center == pytest.approx(math_data["pca"]["center"])

    # Ensure we have as many expected coords as calculated coords.
    assert len(projected_ptpts.index) == len(expected_projected_ptpts)

    for projection in expected_projected_ptpts:
        expected_xy = projection["xy"]
        calculated_xy = projected_ptpts.loc[projection["participant_id"], ["x", "y"]].values

        assert calculated_xy == pytest.approx(expected_xy)

    # Check that the cluster labels all match when K is forced to match.
    _, expected_cluster_labels = extract_data_from_polismath(math_data)
    calculated_cluster_labels = projected_ptpts["cluster_id"].values
    assert_array_equal(calculated_cluster_labels, expected_cluster_labels)

    # from reddwarf.data_presenter import generate_figure
    # print(projected_ptpts)
    # # Flip these to render figure properly.
    # projected_ptpts[["x", "y"]] = -projected_ptpts[["x", "y"]]
    # labels = projected_ptpts["cluster_id"].values
    # generate_figure(projected_ptpts, labels)

# Not used right now. Maybe later.
def groupsort_pids_by_cluster(df):
    """
    Helper function to gather statement IDS in clusters and sort them for easy
    comparison.

    This make comparison easy, even when kmeans gives different numeric labels.

    Args:
        df (pd.DataFrame): A dataframe with projected participants, columns "x",
        "y", "cluster_id"
    Returns:
        (list[list[int]]): A list of lists, each containing statement IDs in a
        cluster.
    """
    # Group by cluster_id and collect indices
    grouped = df.groupby('cluster_id').apply(lambda x: list(x.index))

    # Sort the groups by their length (number of members) in descending order
    sorted_groups = sorted(grouped, key=len, reverse=True)

    # Convert each inner list to integers
    return [list(map(int, group)) for group in sorted_groups]

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering_is_reproducible(polis_convo_data):
    math_data, data_path, _ = polis_convo_data

    # Transpose base cluster coords into participant_ids
    expected_projected_ptpts = transform_base_clusters_to_participant_coords(math_data["base-clusters"])

    client = PolisClient()
    client.load_data(filepaths=[
        f"{data_path}/votes.json",
        # Loading these helps generate mod_out_statement_ids
        f"{data_path}/comments.json",
        f"{data_path}/conversation.json",
    ])

    _, _, mod_out_statement_ids, _ = process_statements(statement_data=client.data_loader.comments_data)

    cluster_run_1 = run_clustering(
        votes=client.data_loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    proj, comps, eigenvals, means, cluster_centers = cluster_run_1 # just to document

    max_group_count = 5
    padded_cluster_centers = pad_to_size(cluster_centers, max_group_count)

    cluster_run_2 = run_clustering(
        votes=client.data_loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        init_centers=padded_cluster_centers,
    )
    proj, comps, eigenvals, means, cluster_centers = cluster_run_2 # just to document

    # same number of clusters
    assert len(cluster_run_1[4]) == len(cluster_run_2[4])

    assert_frame_equal(cluster_run_1[0], cluster_run_2[0]) # projected statements and cluster IDs
    assert cluster_run_1[1].tolist() == cluster_run_2[1].tolist() # components/eigenvectors
    assert cluster_run_1[3].tolist() == cluster_run_2[3].tolist() # statement centers/means
    assert cluster_run_1[4].tolist() == cluster_run_2[4].tolist() # cluster centers