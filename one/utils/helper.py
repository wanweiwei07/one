import numpy as np

def print_arr(name, arr, precision=8):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        s = ", ".join(f"{v:.{precision}f}" for v in arr)
        print(f"{name} = np.array([{s}], dtype=np.float32)")
    elif arr.ndim == 2:
        rows = []
        for row in arr:
            rows.append("[" + ", ".join(f"{v:.{precision}f}" for v in row) + "]")
        s = ", ".join(rows)
        print(f"{name} = np.array([{s}], dtype=np.float32)")
    else:
        raise ValueError("print_arr only supports 1D or 2D arrays")

