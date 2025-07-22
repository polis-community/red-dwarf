_REDUCER_REGISTRY = {}

def register_reducer(name: str):
    def decorator(fn):
        _REDUCER_REGISTRY[name] = fn
        return fn
    return decorator

def get_reducer(name: str, **overrides):
    if name not in _REDUCER_REGISTRY:
        raise ValueError(f"Reducer '{name}' not registered.")
    return _REDUCER_REGISTRY[name](**overrides)

def list_reducers():
    return list(_REDUCER_REGISTRY.keys())