class RedDwarfError(Exception):
    """Base class for other exceptions"""
    pass

def try_import(name: str, extra: str | None = None):
    """
    Offer a more helpful ImportError when optional package groups are missing.
    """
    try:
        return __import__(name)
    except ImportError as e:
        msg = f"Missing optional dependency '{name}'."
        if extra:
            msg += f" Try installing with: pip install red-dwarf[{extra}]"
        raise ImportError(msg) from e