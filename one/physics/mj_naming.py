_auto_counter = {}

def alloc_name(prefix: str) -> str:
    cnt = _auto_counter.get(prefix)
    if cnt is None:
        _auto_counter[prefix] = 1
        return f"{prefix}_0"
    _auto_counter[prefix] = cnt + 1
    return f"{prefix}_{cnt}"

def reset_names():
    _auto_counter.clear()