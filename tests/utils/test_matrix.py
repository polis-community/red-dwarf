import pytest
import pandas as pd

from pathlib import Path
from reddwarf.utils.matrix import deduplicate_votes

@pytest.fixture
def votes_df(polis_convo_data):
    """
    Load votes.csv from the fixture and rename columns for deduplication.
    """
    path = Path(polis_convo_data.data_dir) / "votes.csv"
    df = pd.read_csv(path).rename(columns={
        "timestamp": "modified",
        "comment-id": "statement_id",
        "voter-id": "participant_id",
    })
    return df

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_deduplicate_votes_df(votes_df):
    unique, duplicates = deduplicate_votes(votes=votes_df)

    assert len(votes_df) == 4798
    assert len(unique) == 4789
    assert len(duplicates) == 9
    assert len(votes_df) == len(unique) + len(duplicates)

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_deduplicate_votes_dicts(votes_df):
    votes_dicts = votes_df.to_dict(orient="records")
    unique, duplicates = deduplicate_votes(votes=votes_df)

    assert len(votes_dicts) == 4798
    assert len(unique) == 4789
    assert len(duplicates) == 9
    assert len(votes_dicts) == len(unique) + len(duplicates)

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_deduplicate_votes_df_benchmark(votes_df, benchmark):
    benchmark(deduplicate_votes, votes=votes_df)

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_deduplicate_votes_dicts_benchmark(votes_df, benchmark):
    votes_dicts = votes_df.to_dict(orient="records")
    benchmark(deduplicate_votes, votes=votes_dicts)