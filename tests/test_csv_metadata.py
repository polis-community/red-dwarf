import pytest

from reddwarf.data_loader import Loader


def test_csv_metadata_supplement_preserves_moderation_and_hidden_statements(monkeypatch):
    loader = Loader()
    loader.report_id = "test-report"
    loader.conversation_id = "test-conversation"
    csv_comments = [
        {"statement_id": 0, "moderated": 1, "is_meta": None, "count": None},
        {"statement_id": 1, "moderated": -1, "is_meta": None},
    ]

    def load_csv_comments(_directory_url):
        loader.comments_data = csv_comments

    def load_csv_votes(_directory_url):
        loader.votes_data = []

    def load_api_comments():
        # Public responses can omit moderation fields and rejected statements.
        loader.comments_data = [
            {"statement_id": 0, "moderated": None, "is_meta": False, "count": 20}
        ]

    monkeypatch.setattr(loader, "load_remote_export_data_comments", load_csv_comments)
    monkeypatch.setattr(loader, "load_remote_export_data_votes", load_csv_votes)
    monkeypatch.setattr(loader, "load_api_data_comments", load_api_comments)

    with pytest.warns(UserWarning, match="supplement"):
        loader.load_remote_export_data()

    assert loader.comments_data == [
        {"statement_id": 0, "moderated": 1, "is_meta": False, "count": 20},
        {"statement_id": 1, "moderated": -1, "is_meta": None},
    ]
    assert csv_comments[0]["is_meta"] is None
