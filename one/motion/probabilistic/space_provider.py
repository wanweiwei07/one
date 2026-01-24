import numpy as np
import one.motion.probabilistic.state_space as ssp


class SpaceProvider:

    @classmethod
    def from_box_bounds(cls, lmt_low, lmt_high,
                        collider=None,
                        cd_step_size=None,
                        cache_size=10000):
        space = ssp.RealVectorStateSpace(lmt_low, lmt_high)
        return cls(space, collider,
                   cd_step_size, cache_size)

    def __init__(self, state_space, collider=None,
                 max_edge_step=None, cache_size=10000):
        """is_state_valid: callback function: (state: np.ndarray) -> bool"""
        self.ssp = state_space
        self.collider = collider
        if max_edge_step is not None and max_edge_step <= 0.0:
            raise ValueError("max_edge_step must be positive.")
        self.max_edge_step = max_edge_step
        # caches
        self._collision_cache = {}  # {state_key: bool}
        self._fk_cache = {}  # {state_key: transforms}
        self._cache_size = cache_size

    def is_motion_valid(self, state1, state2):
        dist = self.ssp.distance(state1, state2)
        if dist == 0.0:
            return self._is_collided(state1)
        if self.max_edge_step is None or dist <= self.max_edge_step:
            return not (self._is_collided(state1)
                        or self._is_collided(state2))
        n_steps = int(np.ceil(dist / self.max_edge_step))
        for i in range(n_steps + 1):
            t = i / n_steps
            s = self.ssp.interpolate(state1, state2, t)
            if self._is_collided(s):
                return False
        return True

    def is_state_valid(self, state):
        return not self._is_collided(state)

    def enforce_bounds(self, state):
        return self.ssp.enforce_bounds(state)

    def states_equal(self, state1, state2, tol=1e-4):
        return self.ssp.distance(state1, state2) <= tol

    def clear_cache(self):
        self._collision_cache.clear()

    def _is_collided(self, state):
        cached = self._get_cached(state)
        if cached is not None:
            return cached
        else:
            collided = self.collider.is_collided(state)
            if len(self._collision_cache) >= self._cache_size:
                self._collision_cache.clear()
            key = self._state_to_key(state)
            self._collision_cache[key] = collided
            return collided

    def _get_cached(self, state):
        key = self._state_to_key(state)
        if key in self._collision_cache:
            return self._collision_cache[key]
        else:
            return None

    def _state_to_key(self, state):
        return tuple(np.round(state, decimals=3))
