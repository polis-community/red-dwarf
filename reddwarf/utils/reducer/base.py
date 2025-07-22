from numpy.typing import NDArray
import numpy as np
from reddwarf.exceptions import try_import
from reddwarf.utils.matrix import generate_virtual_vote_matrix
from reddwarf.sklearn.transformers import SparsityAwareCapturer, SparsityAwareScaler
from reddwarf.sklearn.pipeline import PatchedPipeline
from typing import Literal, Optional, Tuple, Union, TYPE_CHECKING, TypeAlias

from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    from pacmap import PaCMAP, LocalMAP
    from sklearn.decomposition import PCA

ReducerType: TypeAlias = Literal["pca", "pacmap", "localmap"]
ReducerModel: TypeAlias = Union["PCA", "PaCMAP", "LocalMAP"]


def get_reducer(
    reducer: ReducerType = "pca",
    n_components: int = 2,
    random_state: Optional[int] = None,
    **reducer_kwargs,
) -> ReducerModel:
    # Setting n_neighbors to None defaults to 10 below 10,000 samples, and
    # slowly increases it according to a formula beyond that.
    # See: https://github.com/YingfanWang/PaCMAP?tab=readme-ov-file#parameters
    DEFAULT_N_NEIGHBORS = None
    match reducer:
        case "pacmap" | "localmap":
            pacmap = try_import("pacmap", extra="alt-algos")

            # Override with default if not set
            n_neighbors = reducer_kwargs.pop("n_neighbors", DEFAULT_N_NEIGHBORS)

            ReducerCls = pacmap.PaCMAP if reducer == "pacmap" else pacmap.LocalMAP
            return ReducerCls(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=n_neighbors,  # type:ignore
                **reducer_kwargs,
            )
        case "pca" | _:
            from sklearn.decomposition import PCA

            return PCA(
                n_components=n_components,
                random_state=random_state,
                **reducer_kwargs,
            )


def run_reducer(
    vote_matrix: NDArray,
    reducer: ReducerType = "pca",
    n_components: int = 2,
    **reducer_kwargs,
) -> Tuple[NDArray, Optional[NDArray], ReducerModel]:
    """
    Process a prepared vote matrix to be imputed and return participant and (optionally) statement data,
    projected into reduced n-dimensional space.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (NDArray): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.
        reducer (Literal["pca", "pacmap", "localmap"]): Dimensionality reduction method to use.

    Returns:
        X_participants (NDArray): A numpy array with n-d coordinates for each projected row/participant.
        X_statements (Optional[NDArray]): A numpy array with n-d coordinates for each projected col/statement.
        reducer_model (ReducerModel): The fitted dimensional reduction sci-kit learn estimator.
    """
    match reducer:
        case "pca":
            pipeline = PatchedPipeline(
                [
                    ("capture", SparsityAwareCapturer()),
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("reduce", get_reducer(reducer, n_components=n_components, **reducer_kwargs)),
                    ("scale", SparsityAwareScaler(capture_step="capture")),
                ]
            )
        case "pacmap" | "localmap":
            pipeline = PatchedPipeline(
                [
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("reduce", get_reducer(reducer, n_components=n_components, **reducer_kwargs)),
                ]
            )

    # Generate projections of participants.
    X_participants = pipeline.fit_transform(vote_matrix)

    if reducer == "pca":
        # Generate projections of statements via virtual vote matrix.
        # This projects unit vectors for each feature/statement into PCA space to
        # understand their placement.
        num_cols = vote_matrix.shape[1]
        n_statements = num_cols
        virtual_vote_matrix = generate_virtual_vote_matrix(n_statements)
        X_statements = pipeline.transform(virtual_vote_matrix)
    else:
        X_statements = None

    reducer_model: ReducerModel = pipeline.named_steps["reduce"]

    return X_participants, X_statements, reducer_model