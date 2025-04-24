import numpy as np
from typing import Optional
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]

def calculate_scaling_factors(X_sparse: Array1D | Array2D) -> Array1D:
    """
    Calculate row-based scaling factors from the sparse vote matrix.

    (Outside estimator so available for re-use.)
    """
    # This allows function to work for 2D (full vote_matrix) and 1D (participant_votes).
    # It essentially nest an 1D matrix in a 2D one.
    X_sparse = np.atleast_2d(X_sparse)

    # The total number of statements in matrix.
    # TODO: This included zero'd out (moderated) statements. Should it?
    _, n_col = X_sparse.shape
    n_total_statements = n_features = n_col

    # List of votes per participant row.
    vote_mask = ~np.isnan(X_sparse)
    n_participant_votes = n_non_missing = np.count_nonzero(vote_mask, axis=1)
    # Ref: https://hyp.is/x6nhItMMEe-v1KtYFgpOiA/gwern.net/doc/sociology/2021-small.pdf
    # Ref: https://github.com/compdemocracy/polis/blob/15aa65c9ca9e37ecf57e2786d7d81a4bd4ad37ef/math/src/polismath/math/pca.clj#L155-L1

    # scaling_factors = np.sqrt(n_features / np.maximum(1, n_non_missing))
    scaling_factors = np.sqrt(n_total_statements / np.maximum(1, n_participant_votes))

    return scaling_factors

class SparsityAwareScaler(BaseEstimator, TransformerMixin):
    """
    Scale projected points (participant or statements) based on sparsity of vote
    matrix, to account for any small number of votes by a participant and
    prevent those participants from bunching up in the center.

    Attributes:
        capture_step (str | int | None): Name or index of the capture step in the pipeline.
        X_sparse (np.ndarray | None): A sparse array with shape (n_features,)
    """
    def __init__(self, capture_step: Optional[str | int] = None, X_sparse: Optional[Array1D | Array2D] = None):
        self.capture_step = capture_step
        self.X_sparse = X_sparse

    # See: https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Suppresses warning caused by fit() not being required before usage in transform().
        tags.requires_fit = False
        return tags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaling_factors = self._calculate_scaling_factors()
        return X * scaling_factors[:, np.newaxis]

    def inverse_transform(self, X):
        scaling_factors = self._calculate_scaling_factors()
        return X / scaling_factors[:, np.newaxis]


    def _get_pipeline_step(self, step):
        """
        Fetch the parent pipeline when available via PatchedPipeline usage.
        """
        parent = getattr(self, "_parent_pipeline", None)
        if parent is None:
            raise RuntimeError(
                f"{self.__class__.__name__} cannot resolve `capture_step={step}` "
                "because it is not being used inside a `PatchedPipeline`. "
                "Either use a `PatchedPipeline` or pass `X_sparse` directly."
            )
        if isinstance(step, str):
            return parent.named_steps[step]
        elif isinstance(step, int):
            return parent.steps[step][1]
        else:
            raise ValueError("`capture_step` must be a string (name) or int (index).")

    def _resolve_X_sparse(self):
        """
        Resolve X_sparse (a sparse vote matrix) from argument or prior capture step.
        """
        if self.X_sparse is not None:
            return self.X_sparse

        capture = self._get_pipeline_step(self.capture_step)
        if not hasattr(capture, "X_captured_"):
            raise AttributeError(
                f"Step '{self.capture_step}' does not contain `.X_captured_`. "
                f"Did you run `fit/transform` on the pipeline?"
            )
        return capture.X_captured_

    def _calculate_scaling_factors(self):
        X_sparse = self._resolve_X_sparse()
        return calculate_scaling_factors(X_sparse=X_sparse)

# TODO: Replace to CaptureMixin and CaptureWrapper.
# See: https://chatgpt.com/c/680a512a-f604-800b-8922-1992a8ddf491
class SparsityAwareCapturer(BaseEstimator, TransformerMixin):
    """
    A passthrough transformer that captures and stores the X it receives in
    `self.X_captured_`. Useful in pipelines where a later step needs access to
    this intermediate result.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.X_captured_ = X  # Store the actual input value
        return X