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