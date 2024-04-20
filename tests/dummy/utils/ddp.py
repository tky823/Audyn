import functools
from typing import Any, Callable


def retry_on_file_not_found(num_retries: int, enable: bool = True) -> Callable:
    """Retry function or method up to num_retries times when FileNotFoundError is detected.

    Args:
        num_retries (int): Maximum retry.
        enable (bool): Whether to activate or deactivate.

    """

    def wrapper(func: Any) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            attempts = 0

            if enable:
                while attempts <= num_retries:
                    try:
                        return func(*args, **kwargs)
                    except FileNotFoundError as e:
                        attempts += 1

                        if attempts > num_retries:
                            raise e
            else:
                return func(*args, **kwargs)

        return _wrapper

    return wrapper
