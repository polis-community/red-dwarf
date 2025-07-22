from typing import TYPE_CHECKING, cast
from sklearn.decomposition import PCA

from reddwarf.exceptions import try_import
from .registry import register_reducer

if TYPE_CHECKING:
    import pacmap as pacmap_module


# Setting n_neighbors to None defaults to 10 below 10,000 samples, and
# slowly increases it according to a formula beyond that.
# See: https://github.com/YingfanWang/PaCMAP?tab=readme-ov-file#parameters
DEFAULT_N_NEIGHBORS = None

@register_reducer("pca")
def make_pca(**kwargs):
    defaults: dict = dict(n_components=2, random_state=42)
    defaults.update(kwargs)
    return PCA(**defaults)

@register_reducer("pacmap")
def make_pacmap(**kwargs) -> "pacmap_module.PaCMAP":
    pacmap = try_import("pacmap", extra="alt-algos")
    if TYPE_CHECKING:
        pacmap = cast("pacmap_module", pacmap)

    defaults: dict = dict(n_components=2, n_neighbors=DEFAULT_N_NEIGHBORS)
    defaults.update(kwargs)
    return pacmap.PaCMAP(**defaults)

@register_reducer("localmap")
def make_localmap(**kwargs) -> "pacmap_module.LocalMAP":
    pacmap = try_import("pacmap", extra="alt-algos")
    if TYPE_CHECKING:
        pacmap = cast("pacmap_module", pacmap)

    defaults: dict = dict(n_components=2, n_neighbors=DEFAULT_N_NEIGHBORS)
    defaults.update(kwargs)
    return pacmap.LocalMAP(**defaults)