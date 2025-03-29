import pytest

def test_agora_import_deprecation_warning():
    with pytest.warns(DeprecationWarning, match=r"'reddwarf.agora' is deprecated"):
        import reddwarf.agora