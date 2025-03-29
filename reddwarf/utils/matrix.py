import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import List, Dict, Optional, Literal, TypeAlias
from reddwarf.exceptions import RedDwarfError


VoteMatrix: TypeAlias = pd.DataFrame

def impute_missing_votes(vote_matrix: VoteMatrix) -> VoteMatrix:
    """
    Imputes missing votes in a voting matrix using column-wise mean. All columns must have at least one vote.

    Reference:
        Small, C. (2021). "Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces."
        Specific highlight: <https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf>

    Args:
        vote_matrix (pd.DataFrame):  A vote matrix DataFrame with `NaN`/`None` values where: \
                                        1. rows are voters, \
                                        2. columns are statements, and \
                                        3. values are votes.

    Returns:
        imputed_matrix (pd.DataFrame): The same vote matrix DataFrame imputing missing values with column mean.
    """
    if vote_matrix.isna().all(axis="rows").any():
        raise RedDwarfError("impute_missing_votes does not support vote matrices containing statement columns with no votes.")

    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_matrix = pd.DataFrame(
        mean_imputer.fit_transform(vote_matrix),
        columns=vote_matrix.columns,
        index=vote_matrix.index,
    )
    return imputed_matrix

def filter_votes(
        votes: List[Dict],
        cutoff: Optional[int] = None,
) -> List[Dict]:
    """
    Filters a list of votes.

    If a `cutoff` is provided, votes are filtered based on either:

    - An `int` representing unix timestamp (ms), keeping only votes before or at that time.
        - Any int above 13_000_000_000 is considered a timestamp.
    - Any other positive or negative `int` is considered an index, reflecting where to trim the time-sorted vote list.
        - positive: filters in votes that many indices from start
        - negative: filters out votes that many indices from end

    Args:
        votes (List[Dict]): An unsorted list of vote records, where each record is a dictionary containing:

            - "participant_id": The ID of the voter.
            - "statement_id": The ID of the statement being voted on.
            - "vote": The recorded vote value.
            - "modified": A unix timestamp object representing when the vote was made.

        cutoff (int): A cutoff unix timestamp (ms) or index position in date-sorted votes list.

    Returns:
        votes (List[Dict]): An list of vote records, sorted by `modified` if index-based filtering occurred.
    """
    if cutoff:
        # TODO: Detect datetime object as arg instead.
        try:
            if cutoff > 1_300_000_000:
                cutoff_timestamp = cutoff
                votes = [v for v in votes if v['modified'] <= cutoff_timestamp]
            else:
                cutoff_index = cutoff
                votes = sorted(votes, key=lambda x: x["modified"])
                votes = votes[:cutoff_index]
        except KeyError as e:
            raise RedDwarfError("The `modified` key is missing from a vote object that must be sorted") from e

    return votes

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
    if cutoff:
        votes = filter_votes(votes=votes, cutoff=cutoff)

    raw_matrix = pd.DataFrame.from_dict(votes)
    raw_matrix = raw_matrix.pivot(
        values="vote",
        index="participant_id",
        columns="statement_id",
    )

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
    """
    for tid in mod_out_statement_ids:
        # Zero out column only if already exists (ie. has votes)
        if tid in vote_matrix.columns:
            vote_matrix.loc[:, tid] = 0

    return vote_matrix

def get_participant_ids(vote_matrix, vote_threshold):
    return vote_matrix[vote_matrix.count(axis="columns") >= vote_threshold].index.to_list()


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
    participant_ids_in = get_participant_ids(vote_matrix, min_user_vote_threshold)
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