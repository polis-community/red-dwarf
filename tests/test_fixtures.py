import pytest
from tests.fixtures import polis_convo_data

from tests.test_data_loader import SMALL_CONVO_ID, SMALL_CONVO_REPORT_ID

@pytest.mark.parametrize("polis_convo_data", [SMALL_CONVO_REPORT_ID], indirect=True)
def test_loading_remote_fixture_from_report_id(polis_convo_data):
    # TODO: Confirm that length is longer than get_clusterable_participant_ids list.
    assert len(polis_convo_data.keep_participant_ids) > 0

@pytest.mark.parametrize("polis_convo_data", [SMALL_CONVO_ID], indirect=True)
def test_loading_remote_fixture_from_convo_id(polis_convo_data):
    # TODO: Confirm that length is longer than get_clusterable_participant_ids list.
    assert len(polis_convo_data.keep_participant_ids) > 0