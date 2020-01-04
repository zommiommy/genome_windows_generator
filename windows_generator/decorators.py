import os
import pandas as pd
from compress_pickle import dump, load

def meta_cache_decorator(path_format, load_cache, store_cache):
    def cache_decorator(f):
        def wrapped_method(self, *args, **kwargs):
            path = path_format.format(**{**vars(self), **kwargs})
            if os.path.exists(path):
                value = load_cache(path)
                return value
            result = f(self, *args, **kwargs)
            os.makedirs(os.path.split(path)[0], exist_ok=True) 
            store_cache(result, path)
            return result
        return wrapped_method
    return cache_decorator

def cache_method(path_format):
    """
    
    """
    if path_format.endswith(".gz") or path_format.endswith(".pkl"):
        return meta_cache_decorator(path_format, load, dump)
    elif path_format.endswith(".csv") or path_format.endswith(".bed"):
        return meta_cache_decorator(
            path_format,
            lambda path: pd.read_csv(path, sep="\t"),
            lambda result, path: result.to_csv(path, sep="\t", index=False)
        )
    else:
        raise ValueError("The path_format [{}] it's not of a known extension".format(path_format))


def multiprocess(f):
    def wrapped(args):
        try:
            return f(*args)
        except KeyboardInterrupt:
            pass
    name = f"multiprocess_decorated_{f.__name__}"
    wrapped.__name__ = name
    wrapped.__qualname__ = name
    globals().update({name:wrapped})
    return wrapped
