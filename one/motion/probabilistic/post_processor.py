import numpy as np


def path_length(path, distance):
    return float(sum(distance(a, b) for a, b in zip(path[:-1], path[1:])))


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

    def tighten(self, path, detour=1.2, n_iter=100, n_partial=60, step=np.pi / 36):
        """Shorten a path that detours: longer than ``detour`` times the straight
        start-goal distance (others come back as they are, at no cost). The path
        is densified to ``step`` so a shortcut may cut mid-edge, shortcut
        ``n_iter`` times, then ``n_partial`` times straightened in one joint at a
        time (partial shortcuts: a joint swinging out and back while the others
        need the detour). Returns the path, still through ``path``'s ends."""
        ctx = self._pln_ctx
        path = [np.asarray(q, dtype=np.float32) for q in path]
        direct = ctx.distance(path[0], path[-1])
        if len(path) <= 2 or path_length(path, ctx.distance) <= detour * max(direct, 1e-9):
            return path
        path = self.shortcut(self.densify(path, max_step=step), n_iter=n_iter)
        path = self.densify(path, max_step=step)
        dim = len(path[0])
        for _ in range(n_partial):
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            d = np.random.randint(dim)
            seg = np.array(path[i:j + 1])
            new = np.linspace(seg[0, d], seg[-1, d], len(seg))
            if np.allclose(new, seg[:, d], atol=1e-4):
                continue
            seg[:, d] = new
            if all(ctx.is_motion_valid(a, b) for a, b in zip(seg[:-1], seg[1:])):
                path[i:j + 1] = list(seg)
        return self.shortcut(path, n_iter=n_iter // 4)
