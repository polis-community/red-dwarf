import sys
import importlib
import builtins
import pytest

from sklearn.datasets import make_blobs


realimport = builtins.__import__

def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    OPTIONAL_MODULE_NAMES = ["hdbscan", "pacmap", "concave_hull", "matplotlib"]

    if name in OPTIONAL_MODULE_NAMES:
        raise ImportError("Simulated missing module")
    return realimport(name, globals, locals, fromlist, level)


def test_missing_extras_pca_works(monkeypatch):
    # Ensure clean re-import
    sys.modules.pop("reddwarf.utils.reducer.base", None)

    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)
        from reddwarf.utils.reducer.base import get_reducer

        get_reducer(reducer="pca")

def test_missing_extras_kmeans_works(monkeypatch):
    # Ensure clean re-import
    sys.modules.pop("reddwarf.utils.clusterer.base", None)

    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)
        from reddwarf.utils.clusterer.base import run_clusterer

        X, *_ = make_blobs(centers=3, cluster_std=0.5, random_state=0)
        run_clusterer(
            X_participants_clusterable=X,
            clusterer="kmeans",
        )

# Test dynamic/runtime import of alternative algorithm modules.

def test_missing_extras_hdbscan_fails(monkeypatch):
    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)
        from reddwarf.utils.clusterer.base import run_clusterer

        with pytest.raises(ImportError) as exc_info:
            X, *_ = make_blobs(centers=3, cluster_std=0.5, random_state=0)
            run_clusterer(
                X_participants_clusterable=X,
                clusterer="hdbscan",
            )

        assert "Missing optional dependency 'hdbscan'" in str(exc_info.value)
        assert "pip install red-dwarf[alt-algos]" in str(exc_info.value)

def test_missing_extras_pacmap_fails(monkeypatch):
    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)
        from reddwarf.utils.reducer.base import get_reducer

        with pytest.raises(ImportError) as exc_info:
            get_reducer(reducer="pacmap")

        assert "Missing optional dependency 'pacmap'" in str(exc_info.value)
        assert "pip install red-dwarf[alt-algos]" in str(exc_info.value)

def test_missing_extras_localmap_fails(monkeypatch):
    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)
        from reddwarf.utils.reducer.base import get_reducer

        with pytest.raises(ImportError) as exc_info:
            get_reducer(reducer="localmap")

        assert "Missing optional dependency 'pacmap'" in str(exc_info.value)
        assert "pip install red-dwarf[alt-algos]" in str(exc_info.value)

# Test static import of matplotlib

def test_missing_extras_data_presenter_fails(monkeypatch):
    # Ensure clean re-import
    import reddwarf.data_presenter
    sys.modules.pop("reddwarf.data_presenter", None)

    with monkeypatch.context() as mp:
        # Patch __import__ only within this block
        mp.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ImportError) as exc_info:
            importlib.import_module("reddwarf.data_presenter")

        assert "Missing optional dependency 'matplotlib'" in str(exc_info.value)
        assert "pip install red-dwarf[plots]" in str(exc_info.value)
