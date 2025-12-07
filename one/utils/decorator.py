import numpy as np
import functools


def mark_dirty(dirty_method_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, value):
            func(self, value)
            getattr(self, dirty_method_name)()
        return wrapper
    return decorator

def lazy_update(dirty_flag_name, update_method_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self):
            if getattr(self, dirty_flag_name):
                getattr(self, update_method_name)()
            return func(self)
        return wrapper
    return decorator

def check_positive(func):
    def wrapper(value, *args, **kwargs):
        if value <= 1e-8:
            raise ValueError(f"{func.__name__}: value must be > 0.")
        return func(value, *args, **kwargs)
    return wrapper

def readonly_view(func):
    def wrapper(self, *args, **kwargs):
        arr = func(self, *args, **kwargs)
        if isinstance(arr, np.ndarray):
            v = arr.view()
            v.flags.writeable = False
            return v
        return arr
    return wrapper