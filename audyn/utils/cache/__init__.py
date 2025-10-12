import os


def get_cache_dir() -> str:
    """Get cache directory for Audyn.

    .. note::

        You can set cache directory by setting ``AUDYN_CACHE_DIR`` environment variable.

    Returns:
        str: Cache directory path.

    """

    _home_dir = os.path.expanduser("~")
    cache_dir = os.getenv("AUDYN_CACHE_DIR") or os.path.join(_home_dir, ".cache", "audyn")

    return cache_dir


def get_model_cache_dir() -> str:
    """Get model cache directory for Audyn.

    Returns:
        str: Cache directory path.

    """

    cache_dir = get_cache_dir()
    model_cache_dir = os.path.join(cache_dir, "models")

    return model_cache_dir
