"""Probabilistic roadmaps.

A ``PRMRoadmap`` is a plain graph of joint states. Planners only read it: the
query's start / goal join it for that query alone, and an edge found in collision
is skipped for that query alone. So one roadmap serves many queries and many
worlds (e.g. the same arm at another base pose, or with the other arm elsewhere),
and can be built once, ``save``d and ``load``ed.

``LazyPRMPlanner`` checks nothing while building beyond the states themselves and
checks a path's states and edges only when Dijkstra picks it. Pass ``world_key``
to ``solve`` to remember those checks across queries in the same world (the
caller promises that equal keys mean equal collision worlds).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import one.motion.probabilistic.post_processor as omppp


class PRMRoadmap:
    def __init__(self):
        self.states = []
        self._edges = {}  # dict of ((i,j): weight), i < j
        self._csr = None
        self._coo = None
        self._kdtree = None

    def __len__(self):
        return len(self.states)

    def add_state(self, q):
        idx = len(self.states)
        self.states.append(np.asarray(q, dtype=np.float32))
        self._kdtree = None
        self._csr = self._coo = None
        return idx

    def add_edge_undirected(self, i, j, w):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        old = self._edges.get((a, b))
        if old is None or w < old:
            self._edges[(a, b)] = float(w)
            self._csr = self._coo = None

    def remove_edge_undirected(self, i, j):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self._edges:
            del self._edges[(a, b)]
            self._csr = self._coo = None

    def knn(self, q, k, exclude_idx=None):
        """(indices, distances) of the ``k`` states nearest ``q``."""
        if self._kdtree is None:
            self._kdtree = cKDTree(np.asarray(self.states, dtype=np.float32))
        k = min(k + (exclude_idx is not None), len(self.states))
        dists, idxs = self._kdtree.query(q, k=k)
        dists, idxs = np.atleast_1d(dists), np.atleast_1d(idxs)
        if exclude_idx is not None:
            keep = idxs != exclude_idx
            dists, idxs = dists[keep], idxs[keep]
        return idxs, dists

    def _edge_coo(self):
        """Both directions of every edge as (rows, cols, weights)."""
        if self._coo is None:
            ab = np.array(list(self._edges.keys()), dtype=np.int64).reshape(-1, 2)
            w = np.fromiter(self._edges.values(), dtype=np.float64, count=len(self._edges))
            w = np.maximum(w, 1e-9)  # an explicit 0 would read as no edge
            self._coo = (np.concatenate([ab[:, 0], ab[:, 1]]),
                         np.concatenate([ab[:, 1], ab[:, 0]]),
                         np.concatenate([w, w]))
        return self._coo

    def _build_csr(self):
        if self._csr is not None:
            return self._csr
        n = len(self.states)
        if n == 0 or len(self._edges) == 0:
            return None
        rows, cols, data = self._edge_coo()
        self._csr = csr_matrix((data, (rows, cols)), shape=(n, n))
        return self._csr

    def query_graph(self, extra):
        """The roadmap as a csr matrix with two more nodes, ``n`` (start) and
        ``n + 1`` (goal), joined by ``extra`` = [(i, j, w)]. A fresh copy: a
        query may block edges in it."""
        n = len(self.states)
        rows, cols, data = self._edge_coo()
        if extra:
            e = np.asarray(extra, dtype=np.float64)
            ei, ej = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
            ew = np.maximum(e[:, 2], 1e-9)
            rows = np.concatenate([rows, ei, ej])
            cols = np.concatenate([cols, ej, ei])
            data = np.concatenate([data, ew, ew])
        return csr_matrix((data, (rows, cols)), shape=(n + 2, n + 2))

    def dijkstra(self, start_idx, goal_idx):
        graph = self._build_csr()
        if graph is None:
            return None
        return _shortest(graph, start_idx, goal_idx)

    def save(self, path):
        ab = np.array(list(self._edges.keys()), dtype=np.int32).reshape(-1, 2)
        w = np.fromiter(self._edges.values(), dtype=np.float32, count=len(self._edges))
        np.savez_compressed(path, states=np.asarray(self.states, dtype=np.float32),
                            edges=ab, weights=w)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        rm = cls()
        rm.states = list(data['states'])
        rm._edges = dict(zip(map(tuple, data['edges'].tolist()), data['weights'].tolist()))
        return rm


def _shortest(graph, start_idx, goal_idx):
    """Node indices of the shortest path, or None."""
    dist, pred = dijkstra(csgraph=graph, directed=True, indices=start_idx,
                          return_predecessors=True)
    if not np.isfinite(dist[goal_idx]):
        return None
    path = [goal_idx]
    while path[-1] != start_idx:
        path.append(int(pred[path[-1]]))
    return path[::-1]


def _block(graph, i, j=None):
    """Remove edge (i, j) from a query graph, or every edge of node i."""
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    if j is None:
        nbrs = indices[indptr[i]:indptr[i + 1]].copy()
        data[indptr[i]:indptr[i + 1]] = np.inf
        for k in nbrs:
            row = slice(indptr[k], indptr[k + 1])
            data[row][indices[row] == i] = np.inf
        return
    for a, b in ((i, j), (j, i)):
        row = slice(indptr[a], indptr[a + 1])
        data[row][indices[row] == b] = np.inf


class PRMPlanner:
    """Roadmap planner. Without a roadmap (``roadmap=`` here or ``rm=`` in
    ``solve``) each ``solve`` builds one in ``pln_ctx``. A plain PRM checks its
    edges while building, so it trusts them while searching."""
    _check_roadmap = False  # check roadmap states / edges while searching

    def __init__(self, pln_ctx, k=15,
                 n_samples=300, max_sample_tries=5000,
                 roadmap=None, k_attach=None, max_worlds=32):
        self._pln_ctx = pln_ctx
        self._path_pp = omppp.PathPostProcessor(self._pln_ctx)
        self.k = int(k)
        self.k_attach = int(k_attach) if k_attach is not None else min(8, self.k + 1)
        self.n_samples = int(n_samples)
        self.max_sample_tries = int(max_sample_tries)
        self.roadmap = roadmap
        self._world_checks = {}  # world_key -> {node int / edge (i, j): valid}
        self._max_worlds = int(max_worlds)

    # ------------------------------------------------------------------ #
    def build_roadmap(self, pln_ctx=None, verbose=False):
        """A roadmap of ``n_samples`` states valid in ``pln_ctx`` (default: the
        planner's), each joined to its ``k`` nearest. Returns it (None if no
        state was found); it does not become ``self.roadmap``."""
        ctx = self._pln_ctx if pln_ctx is None else pln_ctx
        samples = self._sample_valid_states(ctx, self.n_samples, verbose=verbose)
        if not samples:
            return None
        rm = PRMRoadmap()
        for q in samples:
            rm.add_state(q)
        self._connect_knn(rm, ctx)
        if verbose:
            print(f"[PRM] roadmap: {len(rm.states)} states, {len(rm._edges)} edges.")
        return rm

    def _connect_knn(self, rm, ctx):
        Q = rm.states
        for i in range(len(Q)):
            idxs, dists = rm.knn(Q[i], self.k, exclude_idx=i)
            for j, dist in zip(idxs, dists):
                j = int(j)
                if j > i and ctx.is_motion_valid(Q[i], Q[j]):
                    rm.add_edge_undirected(i, j, float(dist))

    def _sample_valid_states(self, ctx, n, verbose=False):
        out = []
        tries = 0
        while len(out) < n and tries < self.max_sample_tries:
            tries += 1
            q = ctx.sample_uniform()
            if ctx.is_state_valid(q):
                out.append(np.asarray(q, dtype=np.float32))
        if verbose:
            print(f"[PRM] sampling: got {len(out)}/{n} valid states in {tries} tries.")
        return out

    # ------------------------------------------------------------------ #
    def solve(self, start, goal, rm=None, verbose=False, world_key=None):
        """Path from ``start`` to ``goal`` (shortcut and densified), or None."""
        pln_ctx = self._pln_ctx
        if world_key is None:
            pln_ctx.clear_cache()
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if not pln_ctx.is_state_valid(start):
            print("[PRM] start is invalid.")
            return None
        if not pln_ctx.is_state_valid(goal):
            print("[PRM] goal is invalid.")
            return None
        if rm is None:
            rm = self.roadmap
        if rm is None:
            rm = self.build_roadmap(verbose=verbose)
            if rm is None:
                print("[PRM] failed to build roadmap.")
                return None
        raw_path = self._search(rm, start, goal, world_key, verbose)
        if raw_path is None:
            print("[PRM] no graph path found.")
            return None
        smooth_path = self._path_pp.shortcut(raw_path)
        return self._path_pp.densify(smooth_path)

    def _checks(self, world_key):
        if world_key is None:
            return {}
        if world_key not in self._world_checks:
            if len(self._world_checks) >= self._max_worlds:
                self._world_checks.pop(next(iter(self._world_checks)))
            self._world_checks[world_key] = {}
        return self._world_checks[world_key]

    def _search(self, rm, start, goal, world_key, verbose):
        """States of the shortest valid roadmap path, start and goal included."""
        ctx = self._pln_ctx
        n = len(rm.states)
        ends = (start, goal)

        def state(i):
            return ends[i - n] if i >= n else rm.states[i]

        extra = [(n, n + 1, ctx.distance(start, goal))]
        for t, q in enumerate(ends):
            idxs, dists = rm.knn(q, self.k_attach)
            extra.extend((n + t, int(j), float(d)) for j, d in zip(idxs, dists))
        graph = rm.query_graph(extra)
        known = self._checks(world_key)  # roadmap states / edges, kept per world
        local = {}  # edges to start / goal, this query only
        for key, ok in known.items():
            if not ok:
                _block(graph, *key) if isinstance(key, tuple) else _block(graph, key)

        def node_ok(i):
            if i >= n or not self._check_roadmap:
                return True
            if i not in known:
                known[i] = ctx.is_state_valid(rm.states[i])
            return known[i]

        def edge_ok(a, b):
            key = (a, b) if a < b else (b, a)
            if key[1] < n and not self._check_roadmap:
                return True
            table = local if key[1] >= n else known
            if key not in table:
                table[key] = ctx.is_motion_valid(state(a), state(b))
            return table[key]

        n_rounds = 0
        while True:
            n_rounds += 1
            idx_path = _shortest(graph, n, n + 1)
            if idx_path is None:
                return None
            bad = next((i for i in idx_path if not node_ok(i)), None)
            if bad is not None:
                _block(graph, bad)
                continue
            bad = next(((a, b) for a, b in zip(idx_path[:-1], idx_path[1:])
                        if not edge_ok(a, b)), None)
            if bad is None:
                if verbose:
                    print(f"[PRM] path of {len(idx_path)} states after {n_rounds} rounds.")
                return [state(i) for i in idx_path]
            _block(graph, *bad)


class LazyPRMPlanner(PRMPlanner):
    """A PRM that checks nothing but the states while building, and a path's
    states and edges only when Dijkstra picks it."""
    _check_roadmap = True

    def _connect_knn(self, rm, ctx):
        Q = rm.states
        for i in range(len(Q)):
            idxs, dists = rm.knn(Q[i], self.k, exclude_idx=i)
            for j, dist in zip(idxs, dists):
                j = int(j)
                if j > i:
                    rm.add_edge_undirected(i, j, float(dist))
