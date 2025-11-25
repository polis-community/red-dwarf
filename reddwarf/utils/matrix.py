import warnings
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal, Tuple, TypeAlias, Union
from reddwarf.exceptions import RedDwarfError


VoteMatrix: TypeAlias = pd.DataFrame

KeepType = Literal["first", "last", False]

def deduplicate_votes(
    votes: Union[pd.DataFrame, List[Dict]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate vote records, keeping the most recent.

    Parameters
    ----------
    votes : pd.DataFrame or list of dict
        Data to deduplicate. Will be sorted by modified date.

    Returns
    -------
    votes_df_unique : pd.DataFrame
    votes_df_duplicates : pd.DataFrame
    """
    votes_df = pd.DataFrame(votes) if isinstance(votes, list) else votes.copy()

    # Check required columns
    required_cols = {"participant_id", "statement_id", "modified"}
    if not required_cols.issubset(votes_df.columns):
        missing = required_cols - set(votes_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    sorted_df = votes_df.sort_values("modified", ascending=True)

    votes_df_unique = sorted_df.drop_duplicates(
        subset=["participant_id", "statement_id"],
        keep="last",
    )

    votes_df_duplicates = sorted_df.loc[~sorted_df.index.isin(votes_df_unique.index)]

    return votes_df_unique, votes_df_duplicates

def filter_votes(
    votes: Union[pd.DataFrame, List[Dict]],
    cutoff: Optional[Union[int, float]] = None,
    time_col: str = "modified",
    skip_timesorting: bool = False,
) -> pd.DataFrame:
    """
    Filter a list or DataFrame of vote records by percent, timestamp, or index.

    Supports three types of cutoffs:
    1. Percent: float between 0.0 and 1.0 representing fraction of earliest votes to keep.
    2. Unix timestamp (ms): keeps only votes with `time_col` <= cutoff.
    3. Index-based: integer position in time-sorted votes list.
       - Positive: keep first `cutoff` votes
       - Negative: remove last `abs(cutoff)` votes

    Args:
        votes: list of dicts or DataFrame of vote records. Must include `time_col`.
        cutoff: cutoff value (percent, timestamp, index)
        time_col: column name to use for timestamp cutoffs and sorting
        skip_timesorting: if True, skip sorting by `time_col`

    Returns:
        pd.DataFrame: Filtered votes, optionally sorted.

    Raises:
        ValueError: If input is invalid or `time_col` is missing.
    """
    # Convert list to DataFrame
    if isinstance(votes, list):
        votes_df = pd.DataFrame(votes)
    elif isinstance(votes, pd.DataFrame):
        votes_df = votes.copy()
    else:
        raise ValueError("`votes` must be a DataFrame or list of dictionaries.")

    if time_col not in votes_df.columns:
        raise ValueError(f"'{time_col}' column not found; cannot perform time-based filtering.")

    # Sort by time_col unless skipped
    if not skip_timesorting:
        votes_df = votes_df.sort_values(time_col).reset_index(drop=True)

    # No cutoff → return sorted DataFrame
    if cutoff is None or cutoff == 1.0:
        return votes_df

    # Percent-based cutoff
    if isinstance(cutoff, float):
        if not (0.0 < cutoff < 1.0):
            raise ValueError("Percent cutoff must be between 0.0 and 1.0.")
        n_keep = max(1, int(len(votes_df) * cutoff))
        return votes_df.iloc[:n_keep]

    # Integer cutoff
    if isinstance(cutoff, int):
        # Timestamp cutoff detection (assume > 13_000_000_000 ms)
        if cutoff > 1_300_000_000:
            return votes_df.loc[votes_df[time_col] <= cutoff]
        # Index-based cutoff
        return votes_df.iloc[:cutoff] if cutoff >= 0 else votes_df.iloc[:cutoff]

    raise ValueError(f"Invalid cutoff type: {type(cutoff)}. Must be int or float.")

def generate_raw_matrix(
        votes: List[Dict],
        cutoff: Optional[int] = None,
) -> VoteMatrix:
    """
    Generates a raw vote matrix from a list of vote records.

    See `filter_votes` method for details of `cutoff` arg.

    Args:
        votes (List[Dict]): An unsorted list of vote records, where each record is a dictionary containing:

            - "participant_id": The ID of the voter.
            - "statement_id": The ID of the statement being voted on.
            - "vote": The recorded vote value.
            - "modified": A unix timestamp object representing when the vote was made.

        cutoff (int): A cutoff unix timestamp (ms) or index position in date-sorted votes list.

    Returns:
        raw_matrix (pd.DataFrame): A full raw vote matrix DataFrame with NaN values where:

            1. rows are voters,
            2. columns are statements, and
            3. values are votes.

            This includes even voters that have no votes, and statements on which no votes were placed.
    """
    # Will just convert into dataframe if no cutoff.
    votes_df = filter_votes(votes=votes, cutoff=cutoff)

    raw_matrix = votes_df.pivot(
        values="vote",
        index="participant_id",
        columns="statement_id",
    )

    # Ensure consistent column ordering regardless of statement_id type
    # Sort numerically if all columns can be converted to integers
    # If not all values can be converted to int, fall back to natural sorting
    try:
        sorted_columns = sorted(raw_matrix.columns, key=lambda x: int(x))
    except (ValueError, TypeError):
        sorted_columns = sorted(raw_matrix.columns)
    raw_matrix = raw_matrix.reindex(columns=sorted_columns)

    # Ensure consistent index
    try:
        sorted_index = sorted(raw_matrix.index, key=lambda x: int(x))
    except (ValueError, TypeError):
        sorted_index = sorted(raw_matrix.index)
    raw_matrix = raw_matrix.reindex(index=sorted_index)

    return raw_matrix

def get_unvoted_statement_ids(vote_matrix: VoteMatrix) -> List[int]:
    """
    A method intended to be piped into a VoteMatrix DataFrame, returning list of unvoted statement IDs.

    See: <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html>

    Args:
        vote_matrix (pd.DataFrame): A pivot of statements (cols), participants (rows), with votes as values.

    Returns:
        unvoted_statement_ids (List[int]): list of statement IDs with no votes.

    Example:

        unused_statement_ids = vote_matrix.pipe(get_unvoted_statement_ids)
    """
    null_column_mask = vote_matrix.isnull().all()
    null_column_ids = vote_matrix.columns[null_column_mask].tolist()

    return null_column_ids

def simple_filter_matrix(
        vote_matrix: VoteMatrix,
        mod_out_statement_ids: list[int] = [],
) -> VoteMatrix:
    """
    The simple filter on the vote_matrix that is used by Polis prior to running PCA.

    Args:
        vote_matrix (VoteMatrix): A raw vote_matrix (with missing values)
        mod_out_statement_ids (list): A list of moderated-out participant IDs to zero out.

    Returns:
        VoteMatrix: Copy of vote_matrix with statements zero'd out
    """
    vote_matrix = vote_matrix.copy()
    for tid in mod_out_statement_ids:
        # Zero out column only if already exists (ie. has votes)
        if tid in vote_matrix.columns:
            # TODO: Add a flag to try np.nan instead of zero.
            vote_matrix.loc[:, tid] = 0

    return vote_matrix

def get_clusterable_participant_ids(vote_matrix: VoteMatrix, vote_threshold: int) -> list:
    """
    Find participant IDs that meet a vote threshold in a vote_matrix.

    Args:
        vote_matrix (VoteMatrix): A raw vote_matrix (with missing values)
        vote_threshold (int): Vote threshold that each participant must meet

    Returns:
        participation_ids (list): A list of participant IDs that meet the threshold
    """
    # TODO: Make this available outside this function? To match polismath output.
    user_vote_counts = vote_matrix.count(axis="columns")
    participant_ids = list(vote_matrix[user_vote_counts >= vote_threshold].index)
    return participant_ids


def filter_matrix(
        vote_matrix: VoteMatrix,
        min_user_vote_threshold: int = 7,
        active_statement_ids: List[int] = [],
        keep_participant_ids: List[int] = [],
        unvoted_filter_type: Literal["drop", "zero"] = "drop",
) -> VoteMatrix:
    """
    Generates a filtered vote matrix from a raw matrix and filter config.

    Args:
        vote_matrix (pd.DataFrame): The [raw] vote matrix.
        min_user_vote_threshold (int): The number of votes a participant must make to avoid being filtered.
        active_statement_ids (List[int]): The statement IDs that are not moderated out.
        keep_participant_ids (List[int]): Preserve specific participants even if below threshold.
        unvoted_filter_type ("drop" | "zero"): When a statement has no votes, it can't be imputed. \
            This determined whether to drop the statement column, or set all the value to zero/pass. (Default: drop)

    Returns:
        filtered_vote_matrix (VoteMatrix): A vote matrix with the following filtered out:

            1. statements without any votes,
            2. statements that have been moderated out,
            3. participants below the vote count threshold,
            4. participants who have not been explicitly selected to circumvent above filtering.
    """
    # Filter out moderated statements.
    vote_matrix = vote_matrix.filter(active_statement_ids, axis='columns')
    # Filter out participants with less than 7 votes (keeping IDs we're forced to)
    # Ref: https://hyp.is/JbNMus5gEe-cQpfc6eVIlg/gwern.net/doc/sociology/2021-small.pdf
    participant_ids_in = get_clusterable_participant_ids(vote_matrix, min_user_vote_threshold)
    # Add in some specific participant IDs for Polismath edge-cases.
    # See: https://github.com/compdemocracy/polis/pull/1893#issuecomment-2654666421
    participant_ids_in = list(set(participant_ids_in + keep_participant_ids))
    vote_matrix = (vote_matrix
        .filter(participant_ids_in, axis='rows')
        # .filter() and .drop() lost the index name, so bring it back.
        .rename_axis("participant_id")
    )

    # This is otherwise the more efficient way, but we want to keep some participant IDs
    # to troubleshoot edge-cases in upsteam Polis math.
    # self.matrix = self.matrix.dropna(thresh=self.min_votes, axis='rows')

    unvoted_statement_ids = vote_matrix.pipe(get_unvoted_statement_ids)

    # TODO: What about statements with no votes? E.g., 53 in oprah. Filter out? zero?
    # Test this on a conversation where it will actually change statement count.
    if unvoted_filter_type == 'drop':
        vote_matrix = vote_matrix.drop(unvoted_statement_ids, axis='columns')
    elif unvoted_filter_type == 'zero':
        vote_matrix[unvoted_statement_ids] = 0

    return vote_matrix

def generate_virtual_vote_matrix(n_statements: int):
    """
    Creates a matrix of virtual participants, each of whom vote agree on a
    single statement, with no other votes. (This is a variation of an "identity
    matrix", with votes going across the diagonal of a full NaN matrix.)
    """
    # Build an basic identity matrix
    virtual_vote_matrix = np.eye(n_statements)

    # Replace 1s with +1 and 0s with NaN
    # TODO: Why does Polis use -1 (disagree) here? is it the same? BUG?
    AGREE_VAL = 1
    MISSING_VAL = np.nan
    virtual_vote_matrix = np.where(virtual_vote_matrix == 1, AGREE_VAL, MISSING_VAL)

    return virtual_vote_matrix