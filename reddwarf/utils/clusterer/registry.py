"""
Clusterer Registry Module

This module provides a registry system for managing clustering algorithms.
It allows for dynamic registration and retrieval of clusterer factory functions, enabling
a plugin-like architecture for different clustering techniques.

The registry supports:
- Decorator-based registration of clusterer factory functions
- Runtime retrieval of registered clusterers with parameter overrides
- Listing of all available clusterer names

Example:
    Register a custom clusterer factory function:

    >>> @register_clusterer('dbscan')
    ... def make_dbscan(**kwargs):
    ...     # Import and configure the clusterer
    ...     defaults = dict(eps=0.5, min_samples=5)
    ...     defaults.update(kwargs)
    ...     return DBSCAN(**defaults)

    Retrieve and instantiate a clusterer:

    >>> clusterer = get_clusterer('dbscan', eps=0.3)
    >>> print(clusterer.eps)
    0.3

    List all available clusterers:

    >>> clusterers = list_clusterers()
    >>> print(clusterers)
    ['dbscan']
"""

from typing import Any, Callable, Dict, List

# Global registry to store clusterer name -> factory function mappings
_CLUSTERER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_clusterer(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a clusterer factory function in the global registry.

    This decorator allows clusterer factory functions to be registered with a string name,
    making them available for dynamic instantiation via get_clusterer(). The registered
    function should accept keyword arguments and return a configured clusterer instance.

    Args:
        name: The string identifier for the clusterer. Must be unique within
              the registry.

    Returns:
        A decorator function that registers the decorated factory function and returns
        it unchanged.

    Raises:
        No explicit exceptions, but will overwrite existing registrations
        with the same name without warning.

    Example:
        >>> @register_clusterer('dbscan')
        ... def make_dbscan(**kwargs):
        ...     from sklearn.cluster import DBSCAN
        ...     defaults = dict(eps=0.5, min_samples=5)
        ...     defaults.update(kwargs)
        ...     return DBSCAN(**defaults)

        >>> # The clusterer factory is now available in the registry
        >>> 'dbscan' in _CLUSTERER_REGISTRY
        True
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _CLUSTERER_REGISTRY[name] = fn
        return fn
    return decorator


def get_clusterer(name: str, **overrides: Any) -> Any:
    """
    Retrieve and instantiate a registered clusterer by name.

    This function looks up a clusterer factory function by its registered name and
    calls it with the provided keyword arguments. This allows for dynamic creation
    of clusterer instances with custom parameters.

    Args:
        name: The string identifier of the clusterer to retrieve.
        **overrides: Keyword arguments to pass to the clusterer factory function.
                    These will be merged with or override any default parameters
                    defined in the factory function.

    Returns:
        An instance of the requested clusterer, created by calling the registered
        factory function with the provided parameters.

    Raises:
        ValueError: If the specified clusterer name is not found in the registry.

    Example:
        >>> # Assuming 'dbscan' clusterer factory is registered
        >>> clusterer = get_clusterer('dbscan', eps=0.3, min_samples=10)
        >>> print(clusterer.eps)
        0.3

        >>> # This will raise ValueError
        >>> clusterer = get_clusterer('nonexistent')
        ValueError: Clusterer 'nonexistent' not registered.
    """
    if name not in _CLUSTERER_REGISTRY:
        raise ValueError(f"Clusterer '{name}' not registered.")
    return _CLUSTERER_REGISTRY[name](**overrides)


def list_clusterers() -> List[str]:
    """
    Get a list of all registered clusterer names.

    This function returns the names of all currently registered clusterers,
    which can be used to discover available clustering algorithms or for
    validation purposes.

    Returns:
        A list of strings containing the names of all registered clusterers.
        The list will be empty if no clusterers have been registered.

    Example:
        >>> # Assuming some clusterers are registered
        >>> clusterers = list_clusterers()
        >>> print(clusterers)
        ['kmeans', 'hdbscan', 'dbscan']

        >>> # Check if a specific clusterer is available
        >>> if 'dbscan' in list_clusterers():
        ...     clusterer = get_clusterer('dbscan')
    """
    return list(_CLUSTERER_REGISTRY.keys())