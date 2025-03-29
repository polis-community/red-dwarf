import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from reddwarf.utils.matrix import VoteMatrix, impute_missing_votes
from typing import Tuple


def run_pca(
        vote_matrix: VoteMatrix,
        n_components: int = 2,
) -> Tuple[ pd.DataFrame, np.ndarray, np.ndarray, np.ndarray ]:
    """
    Process a prepared vote matrix to be imputed and return projected participant data,
    as well as eigenvectors and eigenvalues.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (pd.DataFrame): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.

    Returns:
        projected_data (pd.DataFrame): A dataframe of projected xy coordinates for each `vote_matrix` row/participant.
        eigenvectors (List[List[float]]): Principal `n` components, one per column/statement/feature.
        eigenvalues (List[float]): Explained variance, one per column/statements/feature.
        means (list[float]): Means/centers of column/statements/features.
    """
    imputed_matrix = impute_missing_votes(vote_matrix)

    pca = PCA(n_components=n_components) ## pca is apparently different, it wants
    pca.fit(imputed_matrix) ## .T transposes the matrix (flips it)

    eigenvectors = pca.components_
    eigenvalues = pca.explained_variance_
    # TODO: Why does this need to be inverted to match polismath output? BUG?
    # TODO: Investigate why some numbers are a bit off here.
    #       ANSWER: Because centers are calculated on unfiltered raw matrix for some reason.
    #       means = -raw_vote_matrix.mean(axis="rows")
    means = pca.mean_

    # Project participant vote data onto 2D using eigenvectors.
    # TODO: Determine what exactly we want to be doing here.

    # (1) This is what we used to use:
    # projected_data = pca.transform(imputed_matrix)

    # (2) Perhaps we could be doing this instead: (would need cleanup to clear out NaN values)
    # projected_data = pca.transform(vote_matrix)

    # (3) This is what we'll do for now, as it reproduced Polis calculations exactly:
    # Project data from raw, non-imputed vote_matrix.
    # TODO: Figure out why signs are flipped here after custom projection, unlike pca.transform().
    # Not required for regular pca.transform(), so perhaps a polismath BUG?
    # We fix this in implementations.run_clustering().
    projected_data = [sparsity_aware_project_ptpt(ptpt_votes, pca.components_, pca.mean_) for pid, ptpt_votes in vote_matrix.iterrows()]

    projected_data = pd.DataFrame(projected_data, index=vote_matrix.index, columns=np.asarray(["x", "y"]))
    projected_data.index.name = "participant_id"

    return projected_data, eigenvectors, eigenvalues, means

# This isn't used right now, as we're doing the scaling in the port of `sparcity_aware_project_ptpt()`.
def scale_projected_data(
        projected_data: pd.DataFrame,
        vote_matrix: VoteMatrix
) -> pd.DataFrame:
    """
    Scale projected participant xy points based on vote matrix, to account for any small number of
    votes by a participant and prevent those participants from bunching up in the center.

    Args:
        projected_data (pd.DataFrame): the project xy coords of participants.
        vote_matrix (VoteMatrix): the processed vote matrix data frame, from which to generate scaling factors.

    Returns:
        scaled_projected_data (pd.DataFrame): The coord data rescaled based on participant votes.
    """
    total_active_comment_count = vote_matrix.shape[1]
    participant_vote_counts = vote_matrix.count(axis="columns")
    # Ref: https://hyp.is/x6nhItMMEe-v1KtYFgpOiA/gwern.net/doc/sociology/2021-small.pdf
    # Ref: https://github.com/compdemocracy/polis/blob/15aa65c9ca9e37ecf57e2786d7d81a4bd4ad37ef/math/src/polismath/math/pca.clj#L155-L156
    participant_scaling_coeffs = np.sqrt(total_active_comment_count / participant_vote_counts).values
    # See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    # Reshape scaling_coeffs list to match the shape of projected_data matrix
    participant_scaling_coeffs = np.reshape(participant_scaling_coeffs, (-1, 1))

    return projected_data * participant_scaling_coeffs

def sparsity_aware_project_ptpt(votes, comps, center):
    """
    Projects a sparse vote vector into PCA space while adjusting for sparsity.
    """
    comps = np.array(comps)  # Shape: (2, n_features)
    center = np.array(center)  # Shape: (n_features,)

    # TODO: This included zerod out (moderated) statements. Should it?
    n_cmnts = len(votes)

    ptpt_votes = np.array(votes)
    statement_mask = ~np.isnan(votes)  # Only consider non-null values

    # Extract relevant values
    x_vals = ptpt_votes[statement_mask] - center[statement_mask]  # Centered values
    # TODO: Extend this to work in 3D
    pc1_vals, pc2_vals = comps[:, statement_mask]  # Select only used components

    # Compute dot product projection
    p1 = np.dot(x_vals, pc1_vals)
    p2 = np.dot(x_vals, pc2_vals)

    n_votes = np.count_nonzero(statement_mask)  # Non-null votes count
    scale = np.sqrt(n_cmnts / max(n_votes, 1))

    return scale * np.array([p1, p2])