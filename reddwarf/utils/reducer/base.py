from numpy.typing import NDArray
import numpy as np
from reddwarf.utils.reducer import load_builtins
from reddwarf.utils.reducer.registry import get_reducer
from reddwarf.utils.matrix import generate_virtual_vote_matrix
from reddwarf.sklearn.transformers import SparsityAwareCapturer, SparsityAwareScaler
from reddwarf.sklearn.pipeline import PatchedPipeline
from typing import Optional, Tuple, Union, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pacmap import PaCMAP, LocalMAP
    from sklearn.decomposition import PCA

ReducerModel: TypeAlias = Union["PCA", "PaCMAP", "LocalMAP"]

load_builtins()

# --- Imputer registry ---
from typing import Callable
from sklearn.impute import SimpleImputer, KNNImputer

_IMPUTER_REGISTRY: dict[str, Callable[..., object]] = {}


def register_imputer(name: str):
    """Decorator to register an imputer constructor under a name."""
    def decorator(func: Callable[..., object]):
        _IMPUTER_REGISTRY[name] = func
        return func
    return decorator


def get_imputer(name: str, **kwargs):
    """Retrieve and construct an imputer by name."""
    if name not in _IMPUTER_REGISTRY:
        raise ValueError(f"Unknown imputer '{name}'. Available: {list(_IMPUTER_REGISTRY.keys())}")
    return _IMPUTER_REGISTRY[name](**kwargs)


# --- Built-in imputers ---
@register_imputer("mean")
def make_mean_imputer(**kwargs):
    defaults: dict = dict(missing_values=np.nan, strategy="mean")
    defaults.update(kwargs)
    return SimpleImputer(**defaults)

@register_imputer("knn")
def make_knn_imputer(**kwargs):
    defaults: dict = dict(missing_values=np.nan, n_neighbors=5, weights="distance")
    defaults.update(kwargs)
    return KNNImputer(**defaults)

def run_reducer(
    vote_matrix: NDArray,
    imputer: str | None = None,
    imputer_kwargs: dict = {},
    n_components: int = 2,
    reducer: str = "pca",
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
    imputer = imputer or ("mean" if reducer == "pca" else "knn")

    reducer_kwargs.update(n_components=n_components)
    match reducer:
        case "pca":
            pipeline = PatchedPipeline(
                [
                    ("capture", SparsityAwareCapturer()),
                    ("impute", get_imputer(imputer, **imputer_kwargs)),
                    ("reduce", get_reducer(reducer, **reducer_kwargs)),
                    ("scale", SparsityAwareScaler(capture_step="capture")),
                ]
            )
        # Use this basic unscaled pipeline by default.
        case "pacmap" | "localmap" | _:
            pipeline = PatchedPipeline(
                [
                    ("impute", get_imputer(imputer, **imputer_kwargs)),
                    ("reduce", get_reducer(reducer, **reducer_kwargs)),
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