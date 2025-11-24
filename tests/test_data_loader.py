import pytest
from reddwarf.data_loader import Loader

from tests import helpers
from tests.fixtures import polis_convo_data

# 3 groups, 28 ptpts (24 grouped), 63 statements.
# See: https://pol.is/report/r5hr48j8y8mpcffk7crmk
SMALL_CONVO_ID = "9knpdktubt"
SMALL_CONVO_REPORT_ID = "r5hr48j8y8mpcffk7crmk"

def test_load_data_from_api_conversation():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.conversation_data) > 0

    expected_keys = [
        'topic', 'description', 'is_anon', 'is_active', 'is_draft',
        'is_public', 'email_domain', 'owner', 'participant_count',
        'created', 'strict_moderation', 'profanity_filter', 'spam_filter',
        'context', 'modified', 'owner_sees_participation_stats', 'course_id',
        'link_url', 'upvotes', 'parent_url', 'vis_type', 'write_type',
        'bgcolor', 'help_type', 'socialbtn_type', 'style_btn',
        'auth_needed_to_vote', 'auth_needed_to_write', 'auth_opt_fb',
        'auth_opt_tw', 'auth_opt_allow_3rdparty', 'help_bgcolor',
        'help_color', 'is_data_open', 'is_curated', 'dataset_explanation',
        'write_hint_type', 'subscribe_type', 'org_id', 'need_suzinvite',
        'use_xid_whitelist', 'prioritize_seed', 'importance_enabled',
        'site_id', 'translations', 'treevite_enabled',
        'ownername', 'is_mod', 'is_owner', 'conversation_id',
    ]
    assert sorted(loader.conversation_data) == sorted(expected_keys)

def test_load_data_from_api_comments():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.comments_data) > 0

def test_load_data_from_api_votes():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.votes_data) > 0

def test_load_data_from_api_math():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.math_data) > 0

    # TODO: Test for presences of sub-keys.
    expected_keys = [
        'comment-priorities',
        'user-vote-counts',
        'meta-tids',
        'pca',
        'group-clusters',
        'n',
        'consensus',
        'n-cmts',
        'repness',
        'group-aware-consensus',
        'mod-in',
        'votes-base',
        'base-clusters',
        'mod-out',
        'group-votes',
        'lastModTimestamp',
        'in-conv',
        'tids',
        'lastVoteTimestamp',
        'math_tick',
    ]
    assert sorted(loader.math_data.keys()) == sorted(expected_keys)

def test_load_data_from_api_and_dump_files(tmp_path):
    Loader(conversation_id=SMALL_CONVO_ID, output_dir=str(tmp_path))

    convo_path = tmp_path / "conversation.json"
    votes_path = tmp_path / "votes.json"
    comments_path = tmp_path / "comments.json"
    math_path = tmp_path / "math-pca2.json"

    assert convo_path.exists() == True
    assert votes_path.exists() == True
    assert comments_path.exists() == True
    assert math_path.exists() == True

def test_load_data_from_report_id():
    loader = Loader(report_id="r5hr48j8y8mpcffk7crmk")
    assert len(loader.votes_data) > 0

def test_load_data_from_conversation_id():
    loader = Loader(conversation_id="9knpdktubt")
    assert len(loader.votes_data) > 0

def test_load_data_via_polis_id_report_id():
    report_id = SMALL_CONVO_REPORT_ID
    loader = Loader(polis_id=report_id)

    assert loader.polis_id == report_id
    # Auto-populated from report_id API call.
    assert loader.conversation_id != None
    assert loader.report_id == report_id
    assert len(loader.report_data) > 0

def test_load_data_via_polis_id_convo_id():
    convo_id = SMALL_CONVO_ID
    loader = Loader(polis_id=convo_id)

    assert loader.polis_id == convo_id
    assert loader.conversation_id == convo_id
    # Can't get report_id from convo_id
    assert loader.report_id == None
    assert len(loader.votes_data) > 0

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_load_data_from_file_votes(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])

    assert loader.votes_data != []
    assert len(loader.votes_data) > 0

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_load_data_from_file_comments(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/comments.json"])

    assert loader.comments_data != []
    assert len(loader.comments_data) > 0

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_load_data_from_file_conversation(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/conversation.json"])

    assert loader.conversation_data != {}
    assert len(loader.conversation_data) > 0

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_load_data_from_file_math(polis_convo_data):
    fixture = polis_convo_data
    loader = Loader(filepaths=[f"{fixture.data_dir}/math-pca2.json"])

    assert loader.math_data != {}
    assert len(loader.math_data) > 0

def test_load_data_from_api_report():
    loader = Loader(report_id=SMALL_CONVO_REPORT_ID)
    assert len(loader.report_data) > 0

    expected_keys = [
        'report_id',
        'created',
        'mod_level',
        'modified',
        'label_x_neg',
        'label_y_neg',
        'label_y_pos',
        'label_x_pos',
        'label_group_0',
        'label_group_1',
        'label_group_2',
        'label_group_3',
        'label_group_4',
        'label_group_5',
        'label_group_6',
        'label_group_7',
        'label_group_8',
        'label_group_9',
        'report_name',
        'conversation_id'
    ]
    assert sorted(loader.report_data.keys()) == sorted(expected_keys)

def test_load_data_from_api_with_report_id_only():
    loader = Loader(report_id=SMALL_CONVO_REPORT_ID)
    # Just test one to see if conversation_id determined properly and loaded rest.
    assert len(loader.votes_data) > 0

def test_load_data_from_api_with_report_id_without_conflict():
    loader = Loader(report_id=SMALL_CONVO_REPORT_ID, conversation_id=SMALL_CONVO_ID)
    assert len(loader.votes_data) > 0

def test_load_data_from_api_with_report_id_with_conflict():
    with pytest.raises(ValueError):
        Loader(report_id=SMALL_CONVO_REPORT_ID, conversation_id="conflict-id")

def test_load_data_from_csv_export_without_report_id():
    with pytest.raises(ValueError) as e_info:
        Loader(conversation_id=SMALL_CONVO_ID, data_source="csv_export")
    assert "Cannot determine CSV export URL without report_id or directory_url" == str(e_info.value)

def test_load_data_from_unknown_data_source():
    with pytest.raises(ValueError) as e_info:
        Loader(conversation_id=SMALL_CONVO_ID, data_source="does-not-exist")
    assert "Unknown data_source: does-not-exist" == str(e_info.value)

def test_load_data_from_csv_export_comments():
    loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="csv_export")
    assert len(loader.comments_data) > 0

def test_load_data_from_csv_export_votes():
    loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="csv_export")
    assert len(loader.votes_data) > 0

def test_load_data_from_api_matches_csv_export():
    api_loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="api")
    csv_loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="csv_export")

    assert len(api_loader.comments_data) == len(csv_loader.comments_data)
    assert len(api_loader.votes_data) == len(csv_loader.votes_data)

def test_track_skipped():
    # Should not be dups via API
    api_loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="api")
    assert len(api_loader.skipped_dup_votes) == 0

    # Should be dups via CSV export
    csv_loader = Loader(report_id=SMALL_CONVO_REPORT_ID, data_source="csv_export")
    assert len(csv_loader.skipped_dup_votes) > 0

def test_export_data_csv(tmp_path):
    report_id = SMALL_CONVO_REPORT_ID
    loader = Loader(report_id=report_id)

    output_dir = str(tmp_path)
    loader.export_data(output_dir, format="csv")
    # ... and compare them to the original CSVs:

    for type in helpers.ReportType:
        # first, dowload the original for comparison:
        downloaded = helpers.fetch_csv(type, output_dir, report_id)

        with (
            open(downloaded.name) as f_expected,
            open(f"{output_dir}/{type.value}.csv") as f_actual,
        ):

            expected_lines = f_expected.readlines()
            actual_lines = f_actual.readlines()

            # Compare header line is sorted same
            expected_header = expected_lines[0].strip().split(",")
            actual_header = actual_lines[0].strip().split(",")
            assert expected_header == actual_header, "Headers don't match for {type}"

            # Unfortunately we can't easily compare actual lines:
            #  * The originals have some duplicate entries, so #lines don't match
            #  * the originals aren't always ordered, so lines won't match
            #  * the originals' data seems less accurate than ours, so lines won't match

            # Compare with our own data instead:
            if type == helpers.ReportType.COMMENTS or type == helpers.ReportType.COMMENT_GROUPS:
                assert len(loader.comments_data) == len(actual_lines)-1 # -1 for header
            elif type == helpers.ReportType.VOTES:
                assert len(loader.votes_data) == len(actual_lines)-1 # -1 for header
            elif type == helpers.ReportType.PARTICIPANT_VOTES:
                assert len(loader.math_data["user-vote-counts"]) == len(actual_lines)-1 # -1 for header