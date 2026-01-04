import time
import numpy as np


class RRTTree:
    def __init__(self, r_state):
        """r_state: np.ndarray (root state)"""
        self.states = [np.asarray(r_state, dtype=np.float32)]
        self.parents = [-1]  # root has no parent
        # cache
        self._ssnp = np.asarray(self.states)

    def add_node(self, state, pidx):
        """s: state, pidx: parent index"""
        self.states.append(np.asarray(state, dtype=np.float32))
        self.parents.append(pidx)
        # update cache
        self._ssnp = np.asarray(self.states)
        return len(self.states) - 1

    def nearest(self, state, state_space):
        """return index of nearest state in the tree"""
        dists = state_space.vectorized_distance(state, self._ssnp)
        return int(np.argmin(dists))

    def nearest_state(self, state, state_space):
        idx = self.nearest(state, state_space)
        return self.states[idx]

    def path_from_root(self, idx):
        path = []
        while idx != -1:
            path.append(self.states[idx])
            idx = self.parents[idx]
        path.reverse()
        return path


class RRTConnectPlanner:
    def __init__(self, ssp_provider, step_size=np.pi / 36,
                 goal_bias=0.7):
        self._sspp = ssp_provider
        self.goal_bias = goal_bias
        step = np.asarray(step_size, dtype=float)
        dim = self._sspp.ssp.dim
        if step.size == 1:
            self.max_step = np.full(dim, float(step))  # ← broadcast
        else:
            assert step.size == dim
            self.max_step = step

    def solve(self, start, goal, max_iters=1000,
              time_limit=None, verbose=False):
        start_time = time.time()
        # Two trees: start-tree and goal-tree
        t_start = RRTTree(start)
        t_goal = RRTTree(goal)
        for it in range(max_iters):
            if (time_limit is not None and
                    (time.time() - start_time) > time_limit):
                if verbose:
                    print("[RRTConnect] Time limit exceeded")
                return None
            # Sampling with goal bias
            if np.random.rand() < self.goal_bias:
                rand_state = goal
            else:
                rand_state = self._sspp.ssp.sample_uniform()
            # Extend start-tree towards random sample
            status1, new_idx_start = self._extend_tree(t_start, rand_state)
            if status1 != "trapped":
                # Now try to connect goal-tree toward the new node in start-tree
                new_state = t_start.states[new_idx_start]
                status2, new_idx_goal = self._extend_tree(t_goal, new_state)
                if verbose:
                    print(f"[Iter {it}] status1={status1}, status2={status2}")
                # If trees connected (goal tree reached new_state)
                if status2 == "reached":
                    # Build full path
                    path_start = t_start.path_from_root(new_idx_start)
                    path_goal = t_goal.path_from_root(new_idx_goal)
                    path_goal.reverse()  # from connection point to goal
                    return path_start + path_goal[1:]  # avoid duplicate node
            # Swap the trees every iteration
            t_start, t_goal = t_goal, t_start
        if verbose:
            print("[RRTConnect] No path found")
        return None

    def _steer(self, from_state, to_state):
        delta = to_state - from_state
        step = np.clip(delta, -self.max_step, self.max_step)
        new = from_state + step
        return self._sspp.enforce_bounds(new)

    def _extend_tree(self, tree, tgt_state):
        """return status ("trapped", "advanced", "reached"), last_idx"""
        nearest_idx = tree.nearest(tgt_state, self._sspp.ssp)
        nearest_state = tree.states[nearest_idx]
        new_state = self._steer(nearest_state, tgt_state)
        if not self._sspp.check_motion(nearest_state, new_state):
            return "trapped", nearest_idx  # valid nodes in check motion wasted
        last_idx = tree.add_node(new_state, nearest_idx)
        while True:
            cur_state = tree.states[last_idx]
            if self._sspp.states_equal(cur_state, tgt_state):
                return "reached", last_idx
            delta = tgt_state - cur_state
            if np.all(np.abs(delta) <= self.max_step):
                next_state = tgt_state
                pre_reached = True
            else:
                next_state = self._steer(cur_state, tgt_state)
                pre_reached = False
            if not self._sspp.check_motion(cur_state, next_state):
                break
            last_idx = tree.add_node(next_state, last_idx)
            if pre_reached:
                return "reached", last_idx
        return "advanced", last_idx
