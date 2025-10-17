import pytest
import warnings
from reddwarf.data_loader import Loader
import tempfile
import json
import os


def test_filter_negative_statement_ids_from_json():
    """Test that votes with negative statement IDs are filtered out when loading from JSON files."""
    # Create test data with some negative statement IDs
    test_votes = [
        {
            "participant_id": 1,
            "statement_id": 0,
            "vote": 1,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
        {
            "participant_id": 1,
            "statement_id": -1,  # Negative ID - should be filtered
            "vote": 1,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
        {
            "participant_id": 2,
            "statement_id": 1,
            "vote": -1,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
        {
            "participant_id": 2,
            "statement_id": -5,  # Another negative ID - should be filtered
            "vote": 0,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix="_votes.json", delete=False) as f:
        json.dump(test_votes, f)
        temp_file = f.name

    try:
        # Test that warning is issued
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            loader = Loader(filepaths=[temp_file])

            # Check that warning was issued
            assert len(w) == 1
            assert "Skipped 2 votes with negative comment IDs" in str(w[0].message)
            assert "non-standard Polis instances" in str(w[0].message)

        # Check that only valid votes remain
        assert len(loader.votes_data) == 2

        # Check that negative statement IDs were filtered out
        statement_ids = [vote["statement_id"] for vote in loader.votes_data]
        assert 0 in statement_ids
        assert 1 in statement_ids
        assert -1 not in statement_ids
        assert -5 not in statement_ids

    finally:
        # Clean up
        os.unlink(temp_file)


def test_filter_negative_statement_ids_with_strings():
    """Test that string negative statement IDs are also filtered out."""
    loader = Loader()

    test_votes = [
        {"statement_id": "0", "participant_id": 1, "vote": 1},
        {"statement_id": "-1", "participant_id": 1, "vote": 1},  # Should be filtered
        {"tid": "2", "participant_id": 2, "vote": -1},  # Using 'tid' field name
        {"tid": "-3", "participant_id": 2, "vote": 0},  # Should be filtered
        {
            "comment-id": "4",
            "participant_id": 3,
            "vote": 1,
        },  # Using 'comment-id' field name
        {"comment-id": "-2", "participant_id": 3, "vote": 0},  # Should be filtered
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        filtered_votes = loader._filter_negative_statement_ids(test_votes)

        # Check warning was issued
        assert len(w) == 1
        assert "Skipped 3 votes with negative comment IDs" in str(w[0].message)

    # Check correct votes remain
    assert len(filtered_votes) == 3

    # Check that all remaining votes have non-negative statement IDs
    for vote in filtered_votes:
        statement_id = (
            vote.get("statement_id") or vote.get("tid") or vote.get("comment-id")
        )
        assert int(statement_id) >= 0


def test_filter_negative_statement_ids_helper_method():
    """Test the _filter_negative_statement_ids helper method directly."""
    loader = Loader()

    test_votes = [
        {"statement_id": 0, "participant_id": 1, "vote": 1},
        {"statement_id": -1, "participant_id": 1, "vote": 1},  # Should be filtered
        {"tid": 2, "participant_id": 2, "vote": -1},  # Using 'tid' field name
        {"tid": -3, "participant_id": 2, "vote": 0},  # Should be filtered
        {
            "comment-id": 4,
            "participant_id": 3,
            "vote": 1,
        },  # Using 'comment-id' field name
        {"comment-id": -2, "participant_id": 3, "vote": 0},  # Should be filtered
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        filtered_votes = loader._filter_negative_statement_ids(test_votes)

        # Check warning was issued
        assert len(w) == 1
        assert "Skipped 3 votes with negative comment IDs" in str(w[0].message)

    # Check correct votes remain
    assert len(filtered_votes) == 3

    # Check that all remaining votes have non-negative statement IDs
    for vote in filtered_votes:
        statement_id = (
            vote.get("statement_id") or vote.get("tid") or vote.get("comment-id")
        )
        if statement_id is not None:
            assert statement_id >= 0


def test_no_negative_statement_ids():
    """Test that no warning is issued when there are no negative statement IDs."""
    test_votes = [
        {
            "participant_id": 1,
            "statement_id": 0,
            "vote": 1,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
        {
            "participant_id": 2,
            "statement_id": 1,
            "vote": -1,
            "modified": 1691740361800.0,
            "conversation_id": "test",
        },
    ]

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix="_votes.json", delete=False) as f:
        json.dump(test_votes, f)
        temp_file = f.name

    try:
        # Test that no warning is issued
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            loader = Loader(filepaths=[temp_file])

            # Check that no warning was issued
            assert len(w) == 0

        # Check that all votes remain
        assert len(loader.votes_data) == 2

    finally:
        # Clean up
        os.unlink(temp_file)


def test_empty_votes_data():
    """Test that filtering works correctly with empty votes data."""
    loader = Loader()

    filtered_votes = loader._filter_negative_statement_ids([])
    assert len(filtered_votes) == 0


def test_votes_without_statement_id():
    """Test that votes without statement_id fields are preserved."""
    loader = Loader()

    test_votes = [
        {"participant_id": 1, "vote": 1},  # No statement_id field
        {"participant_id": 2, "vote": -1, "statement_id": None},  # None statement_id
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        filtered_votes = loader._filter_negative_statement_ids(test_votes)

        # No warnings should be issued
        assert len(w) == 0

    # All votes should be preserved
    assert len(filtered_votes) == 2


def test_invalid_statement_id_types():
    """Test that invalid statement_id types are preserved (let model validation handle them)."""
    loader = Loader()

    test_votes = [
        {"statement_id": "not_a_number", "participant_id": 1, "vote": 1},
        {"statement_id": [], "participant_id": 2, "vote": -1},
        {"statement_id": {}, "participant_id": 3, "vote": 0},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        filtered_votes = loader._filter_negative_statement_ids(test_votes)

        # No warnings should be issued for invalid types
        assert len(w) == 0

    # All votes should be preserved (let model validation handle invalid types)
    assert len(filtered_votes) == 3
