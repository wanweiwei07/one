import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import one.motion.probabilistic.post_processor as omppp


class PRMRoadmap:
    def __init__(self):
        self.states = []
        self._edges = {}  # dict of ((i,j): weight)
        self._csr = None

    def add_state(self, q):
        idx = len(self.states)
        self.states.append(np.asarray(q, dtype=np.float32))
        return idx

    def add_edge_undirected(self, i, j, w):
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        old = self._edges.get((a, b))
        if old is None or w < old:
            self._edges[(a, b)] = float(w)
            self._csr = None

    def remove_edge_undirected(self, i, j):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self._edges:
            del self._edges[(a, b)]
            self._csr = None

    def _build_csr(self):
        if self._csr is not None:
            return self._csr
        n = len(self.states)
        if n == 0 or len(self._edges) == 0:
            return None
        rows = []
        cols = []
        data = []
        for (a, b), w in self._edges.items():
            rows.append(a)
            cols.append(b)
            data.append(w)
            rows.append(b)
            cols.append(a)
            data.append(w)
        self._csr = csr_matrix((data, (rows, cols)), shape=(n, n))
        return self._csr

    def dijkstra(self, start_idx, goal_idx):
        graph = self._build_csr()
        if graph is None:
            return None
        dist, pred = dijkstra(
            csgraph=graph, directed=False,
            indices=start_idx, return_predecessors=True)
        if np.isinf(dist[goal_idx]):
            return None
        path = []
        cur = goal_idx
        while cur != -9999:
            path.append(cur)
            if cur == start_idx:
                return path[::-1]
            cur = pred[cur]
        return None


class PRMPlanner:
    def __init__(self, pln_ctx, k=15,
                 n_samples=300, max_sample_tries=5000):
        self._pln_ctx = pln_ctx
        self._path_pp = omppp.PathPostProcessor(self._pln_ctx)
        self.k = int(k)
        self.n_samples = int(n_samples)
        self.max_sample_tries = int(max_sample_tries)
        # KD-tree related
        self._Q = None
        self._kdtree = None

    def solve(self, start, goal, rm=None, verbose=False):
        pln_ctx = self._pln_ctx
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
            rm = self._build_roadmap()
            if rm is None:
                print("[PRM] failed to build roadmap.")
                return None
        s_idx = rm.add_state(start)
        g_idx = rm.add_state(goal)
        # rebuild KD-tree because nodes changed
        self._build_kdtree(rm, verbose=verbose)
        self._connect_node_to_roadmap(rm, s_idx)
        self._connect_node_to_roadmap(rm, g_idx)
        # 5) shortest path
        idx_path = rm.dijkstra(s_idx, g_idx)
        if idx_path is None:
            print("[PRM] no graph path found.")
            return None
        raw_path = [rm.states[i] for i in idx_path]
        smooth_path = self._path_pp.shortcut(raw_path)
        return self._path_pp.densify(smooth_path)

    def _build_roadmap(self, verbose=False):
        rm = PRMRoadmap()
        # 1) sample valid states
        samples = self._sample_valid_states(
            self.n_samples, verbose=verbose)
        if not samples:
            return None
        for q in samples:
            rm.add_state(q)
        # 2) build KD-tree on roadmap samples
        self._build_kdtree(rm)
        # 3) connect roadmap (kNN)
        self._connect_knn(rm)
        return rm

    def _build_kdtree(self, rm):
        self._Q = np.asarray(rm.states, dtype=np.float32)
        self._kdtree = cKDTree(self._Q)

    def _query_knn(self, qi, k, exclude_idx):
        """KD-tree kNN query, excluding a specific index"""
        n = len(self._Q)
        k = min(k, n)
        dists, idxs = self._kdtree.query(qi, k=k, workers=-1)
        # ensure iterable
        if np.isscalar(idxs):
            idxs = np.array([idxs])
            dists = np.array([dists])
        if exclude_idx is not None:
            mask = idxs != exclude_idx
            idxs = idxs[mask]
            dists = dists[mask]
        return idxs, dists

    def _connect_knn(self, rm):
        pln_ctx = self._pln_ctx
        Q = self._Q
        n = len(Q)
        if n <= 1:
            return
        for i in range(n):
            qi = Q[i]
            idxs, dists = self._query_knn(
                qi, self.k + 1, exclude_idx=i)
            for j, dist in zip(idxs, dists):
                j = int(j)
                if j <= i:
                    continue
                if pln_ctx.is_motion_valid(qi, Q[j]):
                    rm.add_edge_undirected(
                        i, j, float(dist))

    def _connect_node_to_roadmap(self, rm, node_idx):
        pln_ctx = self._pln_ctx
        Q = self._Q
        qi = Q[node_idx]
        idxs, dists = self._query_knn(
            qi, min(8, self.k + 1), exclude_idx=node_idx)
        for j, dist in zip(idxs, dists):
            if pln_ctx.is_motion_valid(qi, Q[j]):
                rm.add_edge_undirected(
                    node_idx, int(j), float(dist))

    def _sample_valid_states(self, n, verbose=False):
        pln_ctx = self._pln_ctx
        out = []
        tries = 0
        while (len(out) < n and
               tries < self.max_sample_tries):
            tries += 1
            q = pln_ctx.sample_uniform()
            if pln_ctx.is_state_valid(q): # TODO: batch?
                out.append(np.asarray(q, dtype=np.float32))
            if verbose:
                print(f"[PRM] sampling: got {len(out)}/{n} "
                      f"valid states in {tries} tries.")
        return out


class LazyPRMPlanner(PRMPlanner):

    def _connect_knn(self, rm):
        Q = self._Q
        n = len(Q)
        if n <= 1:
            return
        for i in range(n):
            qi = Q[i]
            idxs, dists = self._query_knn(qi, self.k + 1, exclude_idx=i)
            for j, dist in zip(idxs, dists):
                j = int(j)
                if j <= i:
                    continue
                rm.add_edge_undirected(i, j, float(dist))

    def solve(self, start, goal, rm=None, verbose=False):
        pln_ctx = self._pln_ctx
        pln_ctx.clear_cache()
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if not pln_ctx.is_state_valid(start):
            print("[LazyPRM] start is invalid.")
            return None
        if not pln_ctx.is_state_valid(goal):
            print("[LazyPRM] goal is invalid.")
            return None
        if rm is None:
            rm = self._build_roadmap(verbose=verbose)
            if rm is None:
                print("[LazyPRM] failed to build roadmap.")
                return None
        # attach start / goal
        s_idx = rm.add_state(start)
        g_idx = rm.add_state(goal)
        self._build_kdtree(rm)
        self._connect_node_to_roadmap(rm, s_idx)
        self._connect_node_to_roadmap(rm, g_idx)
        Q = np.asarray(rm.states, dtype=np.float32)
        # Lazy validation loop
        while True:
            idx_path = rm.dijkstra(s_idx, g_idx)
            if idx_path is None:
                print("[LazyPRM] no graph path found.")
                return None
            bad_edge = None
            for a, b in zip(idx_path[:-1], idx_path[1:]):
                if not pln_ctx.is_motion_valid(Q[a], Q[b]):
                    bad_edge = (a, b)
                    break
            if bad_edge is None:
                raw_path = [rm.states[i] for i in idx_path]
                smooth_path = self._path_pp.shortcut(raw_path)
                return self._path_pp.densify(smooth_path)
            rm.remove_edge_undirected(*bad_edge)
