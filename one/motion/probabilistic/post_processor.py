import numpy as np


class PathPostProcessor:
    def __init__(self, pln_ctx=None):
        self._pln_ctx = pln_ctx

    def shortcut(self, path, n_iter=200):
        path = list(path)
        for _ in range(n_iter):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if self._pln_ctx.is_motion_valid(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    def densify(self, path, max_step=np.pi / 12):
        dense = [path[0]]
        for q0, q1 in zip(path[:-1], path[1:]):
            dist = self._pln_ctx.distance(q0, q1)
            n = max(1, int(np.ceil(dist / max_step)))
            for i in range(1, n + 1):
                t = i / n
                q = self._pln_ctx.interpolate(q0, q1, t)
                dense.append(q)
        return dense