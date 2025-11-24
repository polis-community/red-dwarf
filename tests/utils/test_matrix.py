import json
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

@pytest.mark.parametrize("polis_convo_data", ["medium-with-meta"], indirect=True)
def test_deduplicate_votes(votes_df):
    deduped, skipped = deduplicate_votes(votes_df=votes_df)

    assert len(votes_df) == 4798
    assert len(deduped) == 4789
    assert len(skipped) == 9
    assert len(votes_df) == len(deduped) + len(skipped)

@pytest.mark.parametrize("polis_convo_data", ["medium-with-meta"], indirect=True)
def test_deduplicate_votes_benchmark(votes_df, benchmark):
    benchmark(deduplicate_votes, votes_df=votes_df)