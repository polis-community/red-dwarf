"""
Reducer Registry Module

This module provides a registry system for managing dimensionality reduction algorithms.
It allows for dynamic registration and retrieval of reducer factory functions, enabling
a plugin-like architecture for different reduction techniques.

The registry supports:
- Decorator-based registration of reducer factory functions
- Runtime retrieval of registered reducers with parameter overrides
- Listing of all available reducer names

Example:
    Register a custom reducer factory function:

    >>> @register_reducer('umap')
    ... def make_umap(**kwargs):
    ...     # Import and configure the reducer
    ...     defaults = dict(n_components=2, n_neighbors=15)
    ...     defaults.update(kwargs)
    ...     return umap.UMAP(**defaults)

    Retrieve and instantiate a reducer:

    >>> reducer = get_reducer('umap', n_components=3)
    >>> print(reducer.n_components)
    3

    List all available reducers:

    >>> reducers = list_reducers()
    >>> print(reducers)
    ['umap']
"""

from typing import Any, Callable, Dict, List

# Global registry to store reducer name -> factory function mappings
_REDUCER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_reducer(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a reducer factory function in the global registry.

    This decorator allows reducer factory functions to be registered with a string name,
    making them available for dynamic instantiation via get_reducer(). The registered
    function should accept keyword arguments and return a configured reducer instance.

    Args:
        name: The string identifier for the reducer. Must be unique within
              the registry.

    Returns:
        A decorator function that registers the decorated factory function and returns
        it unchanged.

    Raises:
        No explicit exceptions, but will overwrite existing registrations
        with the same name without warning.

    Example:
        >>> @register_reducer('umap')
        ... def make_umap(**kwargs):
        ...     defaults = dict(n_components=2, n_neighbors=15)
        ...     defaults.update(kwargs)
        ...     return umap.UMAP(**defaults)

        >>> # The reducer factory is now available in the registry
        >>> 'umap' in _REDUCER_REGISTRY
        True
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REDUCER_REGISTRY[name] = fn
        return fn
    return decorator


def get_reducer(name: str, **overrides: Any) -> Any:
    """
    Retrieve and instantiate a registered reducer by name.

    This function looks up a reducer factory function by its registered name and
    calls it with the provided keyword arguments. This allows for dynamic creation
    of reducer instances with custom parameters.

    Args:
        name: The string identifier of the reducer to retrieve.
        **overrides: Keyword arguments to pass to the reducer factory function.
                    These will be merged with or override any default parameters
                    defined in the factory function.

    Returns:
        An instance of the requested reducer, created by calling the registered
        factory function with the provided parameters.

    Raises:
        ValueError: If the specified reducer name is not found in the registry.

    Example:
        >>> # Assuming 'umap' reducer factory is registered
        >>> reducer = get_reducer('umap', n_components=3, n_neighbors=20)
        >>> print(reducer.n_components)
        3

        >>> # This will raise ValueError
        >>> reducer = get_reducer('nonexistent')
        ValueError: Reducer 'nonexistent' not registered.
    """
    if name not in _REDUCER_REGISTRY:
        raise ValueError(f"Reducer '{name}' not registered.")
    return _REDUCER_REGISTRY[name](**overrides)


def list_reducers() -> List[str]:
    """
    Get a list of all registered reducer names.

    This function returns the names of all currently registered reducers,
    which can be used to discover available reduction algorithms or for
    validation purposes.

    Returns:
        A list of strings containing the names of all registered reducers.
        The list will be empty if no reducers have been registered.

    Example:
        >>> # Assuming some reducers are registered
        >>> reducers = list_reducers()
        >>> print(reducers)
        ['pca', 'tsne', 'umap']

        >>> # Check if a specific reducer is available
        >>> if 'umap' in list_reducers():
        ...     reducer = get_reducer('umap')
    """
    return list(_REDUCER_REGISTRY.keys())
