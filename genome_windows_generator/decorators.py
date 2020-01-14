import os
from compress_pickle import dump, load


def cache_method(path_format):
    def cache_decorator(f):
        def wrapped_method(self, *args, **kwargs):
            path = path_format.format(**{**vars(self), **kwargs})
            if os.path.exists(path):
                value = load(path)
                return value
            result = f(self, *args, **kwargs)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            dump(result, path)
            return result
        return wrapped_method
    return cache_decorator


def multiprocess(f):
    def wrapped(args):
        return f(*args)
    name = f"multiprocess_decorated_{f.__name__}"
    wrapped.__name__ = name
    wrapped.__qualname__ = name
    globals().update({name: wrapped})
    return wrapped
