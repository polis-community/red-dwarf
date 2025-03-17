from reddwarf import utils
from tests.fixtures import polis_convo_data
import pytest

from reddwarf.polis import PolisClient
from reddwarf.types.polis import PolisRepness

def get_grouped_statement_ids(repness: PolisRepness) -> dict[str, list[dict[str, list[int]]]]:
    """A helper to compare only tid in groups, rather than full repness object."""
    groups = []

    for key, statements in repness.items():
        group = {"id": str(key), "members": sorted([stmt["tid"] for stmt in statements])} # type:ignore
        groups.append(group)

    return {"groups": groups}

# TODO: Investigate why "small-with-meta" and "medium" won't pass.
@pytest.mark.parametrize("polis_convo_data", ["small", "small-no-meta"], indirect=True)
def test_calculate_representativeness_real_data(polis_convo_data):
    math_data, path, _ = polis_convo_data
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])

    all_clustered_participant_ids, cluster_labels = utils.extract_data_from_polismath(math_data)

   # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    client.keep_participant_ids = all_clustered_participant_ids
    vote_matrix = client.get_matrix(is_filtered=True)

    # Generate stats all groups and all statements.
    grouped_stats_df, gac_df = utils.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )

    polis_repness = utils.select_representative_statements(grouped_stats_df=grouped_stats_df)

    actual_repness: PolisRepness = polis_repness # type:ignore
    expected_repness: PolisRepness = math_data["repness"] # type:ignore
    # Compare the selected statements calculated from those generated by polismath.
    assert get_grouped_statement_ids(actual_repness) == get_grouped_statement_ids(expected_repness)

    # Cycle through all the expected data calculated by Polis platform
    for group_id, statements in math_data['repness'].items():
        group_id = int(group_id)
        for st in statements:
            expected_repr = st["repness"]
            expected_repr_test = st["repness-test"]
            expected_prob = st["p-success"]
            expected_prob_test = st["p-test"]

            # Fetch matching calculated values for comparison.
            keys = ["prob", "prob_test", "repness", "repness_test"]
            if st["repful-for"] == "agree":
                key_map = dict(zip(keys, ["pa", "pat", "ra", "rat"]))
            else: # disagree
                key_map = dict(zip(keys, ["pd", "pdt", "rd", "rdt"]))

            actual = {
                k: grouped_stats_df[group_id].loc[st["tid"], v]
                for k,v in key_map.items()
            }

            assert actual["prob"] == pytest.approx(expected_prob)
            assert actual["prob_test"] == pytest.approx(expected_prob_test)
            assert actual["repness"] == pytest.approx(expected_repr)
            assert actual["repness_test"] == pytest.approx(expected_repr_test)
