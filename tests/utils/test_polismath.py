import pytest
from reddwarf.utils import polismath
from tests.fixtures import polis_convo_data
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.polis import PolisClient

# TODO: Figure out if in-conv is always equivalent. If so, remove this test.
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_extract_data_from_polismath(polis_convo_data):
    math_data, _, _ = polis_convo_data
    expected = math_data["in-conv"]

    actual_participant_ids, *_ = polismath.extract_data_from_polismath(math_data)

    assert sorted(expected) == actual_participant_ids

# This check is to ensure that the vote_data used to assemble the vote_matrix
# matches what polismath is calculating from. Sometimes votes from random
# participants just don't get returned via API.
#
# Any dataset used to testing should be parameterized here.
#
# Example: participant 50 has 4 votes in exports CSVs and polismath user-vote-count data:
#   https://pol.is/api/v3/math/pca2?conversation_id=6ukkcvfbre
# But no votes show up here:
#   https://pol.is/api/v3/votes?conversation_id=6ukkcvfbre&pid=50
#
# These conversations also have data integrity issues:
#   - https://pol.is/report/r64ajcsmp9butzxhzj44c ("small-no-meta", see above)
#   - https://pol.is/report/r5ackfcj5dmawnt2de7x7
#   - https://pol.is/report/r45ru4zfmutun54ttskne
#
# TODO: Investigate what causes conversations to have missing votes.
#       For now, leaning on other conversations or CSV export mitigates the issue.
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"], indirect=True)
def test_user_vote_count_sanity_check(polis_convo_data):
    math_data, data_path, *_ = polis_convo_data
    expected_user_vote_counts = math_data["user-vote-counts"]
    # Convert keys from string to int for later comparison.
    expected_user_vote_counts = {int(k):v for k,v in expected_user_vote_counts.items()}

    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=client.data_loader.votes_data)
    actual_user_vote_counts = real_vote_matrix.count(axis="columns")

    assert expected_user_vote_counts == actual_user_vote_counts.to_dict()