from numpy.typing import ArrayLike
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
    means = -pca.mean_

    # Project participant vote data onto 2D using eigenvectors.
    projected_data = pca.transform(imputed_matrix)
    projected_data = pd.DataFrame(projected_data, index=imputed_matrix.index, columns=np.asarray(["x", "y"]))
    projected_data.index.name = "participant_id"

    return projected_data, eigenvectors, eigenvalues, means

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

# TODO: Clean up variables and docs.
def sparsity_aware_project_ptpt(participant_votes, statement_components, statement_means):
    """
    Projects a sparse vote vector into PCA space while adjusting for sparsity.

    Args:
        participant_votes: List of participant votes on each statement
        statement_components (list[list[float]]): Two lists of floats corresponding to the two principal components
        statement_means (list[float]): List of floats corresponding to the centers/means of each statement

    Returns:
        projected_coords (list[list[float]]): Two lists corresponding to projected xy coordinates.
    """
    statement_components = np.array(statement_components)  # Shape: (2, n_features)
    statement_means = np.array(statement_means)  # Shape: (n_features,)

    n_statements = len(participant_votes)

    participant_votes = np.array(participant_votes)
    mask = ~np.isnan(participant_votes)  # Only consider non-null values

    # Extract relevant values
    x_vals = participant_votes[mask] - statement_means[mask]  # Centered values
    pc1_vals, pc2_vals = statement_components[:, mask]  # Select only used components

    # Compute dot product projection
    p1 = np.dot(x_vals, pc1_vals)
    p2 = np.dot(x_vals, pc2_vals)

    n_votes = np.count_nonzero(mask)  # Non-null votes count
    scale = np.sqrt(n_statements / max(n_votes, 1))

    projected_coord = scale * np.array([p1, p2])

    return projected_coord

# TODO: Clean up variables and docs.
def sparsity_aware_project_ptpts(vote_matrix, statement_components, statement_means):
    """
    Apply sparsity-aware projection to multiple vote vectors.
    """
    return np.array([
        sparsity_aware_project_ptpt(participant_votes, statement_components, statement_means)
        for participant_votes in vote_matrix]
    )

# TODO: Clean up variables and docs.
def pca_project_cmnts(statement_components, statement_means):
    """
    Projects unit vectors for each feature into PCA space to understand their placement.
    """
    n_statements = len(statement_means)
    # Create a matrix of virtual participants that each vote once on a single statement.
    virtual_vote_matrix = np.full(shape=[n_statements, n_statements], fill_value=np.nan)
    for i in range(n_statements):
        # TODO: Why does Polis use -1 (disagree) here? is it the same? BUG?
        virtual_vote_matrix[i][i] = -1  # Create unit vector representation

    # 40 xy pairs. shape (40, 2)
    statement_projections = sparsity_aware_project_ptpts(
        virtual_vote_matrix,
        statement_components,
        statement_means,
    )

    return statement_projections

def calculate_extremity(projections: ArrayLike):
    # Compute extremity as vector magnitude on rows.
    # vector magnitude = Euclidean norm = hypotenuse of xy
    return np.linalg.norm(projections, axis=0)

# TODO: Clean up variables and docs.
def with_proj_and_extremity(pca):
    """
    Compute projection and extremity, then merge into PCA results.
    """
    statement_projections = pca_project_cmnts(
        statement_components=pca["comps"],
        statement_means=pca["center"],
    )
    # Flip the axes to get all x together and y together.
    # 2 sets of 40. shape (2, 40)
    statement_projections = statement_projections.transpose()

    statement_extremities = calculate_extremity(statement_projections)

    pca["comment-projection"] = statement_projections.tolist()
    pca["comment-extremity"] = statement_extremities.tolist()

    return pca