import numpy as np
import one.motion.core.state_space as ssp


class PlanningContext:
    """Collision + bounds oracle for sampling-based planners.

    Thin and stateless w.r.t. the world: the caller owns the collider. The
    mechanisms being planned are exactly ``collider.actors``; any other body
    (static obstacles, a held gripper at a fixed width, etc.) is posed by the
    caller directly on the collider via ``collider.set_mecba_qpos(...)``,
    followed by ``clear_cache()`` so memoized collisions don't go stale.
    """

    def __init__(self,
                 collider,                  # mj_collider; collider.actors = planned mechanisms
                 joint_limits=None,         # (low, high); autoinfer from actors if None
                 cd_step_size=np.pi / 180,  # edge collision-check step size
                 cache_size=10000):         # collision cache capacity
        if not collider.actors:
            raise ValueError("collider has no actors; set collider.actors first")
        self.collider = collider
        self.cd_step_size = cd_step_size
        if joint_limits is None:
            joint_limits = self._infer_joint_limits()
        self.state_space = ssp.RealVectorStateSpace(*joint_limits)
        self._collision_cache = {}
        self._cache_size = cache_size

    def _infer_joint_limits(self):
        lows, highs = [], []
        for mecba in self.collider.actors:
            compiled = mecba.structure.compiled
            lows.append(compiled.jlmt_low_by_idx)
            highs.append(compiled.jlmt_high_by_idx)
        return np.concatenate(lows), np.concatenate(highs)

    def is_state_valid(self, state):
        if not self.state_space.satisfies_bounds(state):
            return False
        return not self._is_collided(state)

    def is_motion_valid(self, state1, state2):
        # Validate every interpolated step with is_state_valid (not _is_collided
        # directly) so subclass overrides of is_state_valid -- e.g. extra
        # constraints layered on top of collision -- also gate edges, not just
        # nodes. is_state_valid already includes the bounds + collision checks.
        if not self.is_state_valid(state1) or not self.is_state_valid(state2):
            return False
        dist = self.state_space.distance(state1, state2)
        if dist <= self.cd_step_size:
            return True
        n_steps = int(np.ceil(dist / self.cd_step_size))
        for i in range(1, n_steps):
            s = self.state_space.interpolate(state1, state2, i / n_steps)
            if not self.is_state_valid(s):
                return False
        return True

    def states_equal(self, state1, state2, tol=1e-4):
        return self.state_space.distance(state1, state2) <= tol

    def enforce_bounds(self, state):
        return self.state_space.enforce_bounds(state)

    def clear_cache(self):
        self._collision_cache.clear()

    def sample_uniform(self):
        return self.state_space.sample_uniform()

    def interpolate(self, state1, state2, t):
        return self.state_space.interpolate(state1, state2, t)

    def distance(self, state1, state2):
        return self.state_space.distance(state1, state2)

    def _is_collided(self, state):
        key = self._state_to_key(state)
        if key in self._collision_cache:
            return self._collision_cache[key]
        collided = self.collider.is_collided(state)
        if len(self._collision_cache) >= self._cache_size:
            self._collision_cache.clear()
        self._collision_cache[key] = collided
        return collided

    def _state_to_key(self, state):
        return tuple(np.round(state, decimals=3))
