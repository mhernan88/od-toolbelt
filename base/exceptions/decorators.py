from functools import wraps


def log_exception(l):
    def inner_log_exception(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                l.error(e.__str__())
        return wrapper
    return inner_log_exception

