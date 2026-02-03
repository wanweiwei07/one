import numpy as np
import one.motion.probabilistic.state_space as ssp

class PlanningContext:

    def __init__(self,
                 collider,  # mj_collider instance
                 planning_mecbas=None,  # TODO delete
                 aux_mecbas=None,  # TODO delete
                 joint_limits=None,  # autoinfer if None
                 cd_step_size=np.pi / 180,  # edge collision check step size
                 cache_size=10000):  # collision cache size
        self.collider = collider
        if planning_mecbas is None:
            if not collider.actors:
                raise ValueError("No planning mecbas provided")
            self.planning_mecbas = tuple(collider.actors)
        else:
            self.planning_mecbas = tuple(planning_mecbas)
            collider.actors = list(self.planning_mecbas)
        self.aux_mecbas = aux_mecbas or {}
        self.cd_step_size = cd_step_size
        if joint_limits is None:
            lmt_low, lmt_high = self._infer_joint_limits()
        else:
            lmt_low, lmt_high = joint_limits
        self.state_space = ssp.RealVectorStateSpace(lmt_low, lmt_high)
        for mecba, qs in self.aux_mecbas.items():
            collider.set_mecba_qpos(mecba, qs)
        # cache for collision checking
        self._collision_cache = {}
        self._cache_size = cache_size

    def _infer_joint_limits(self):
        lmt_low_list = []
        lmt_high_list = []
        for mecba in self.planning_mecbas:
            compiled = mecba.structure.compiled
            lmt_low_list.append(compiled.jlmt_low_by_idx)
            lmt_high_list.append(compiled.jlmt_high_by_idx)
        return np.concatenate(lmt_low_list), np.concatenate(lmt_high_list)

    def set_aux_mecbas(self, mecba, qs):
        if mecba in self.planning_mecbas:
            raise ValueError("Cannot set state of planning mecba")
        self.aux_mecbas[mecba] = qs
        self.collider.set_mecba_qpos(mecba, qs)
        self._collision_cache.clear()

    def is_state_valid(self, state):
        return not self._is_collided(state)

    def is_motion_valid(self, state1, state2):
        dist = self.state_space.distance(state1, state2)
        if dist == 0.0:
            return not self._is_collided(state1)
        if dist <= self.cd_step_size:
            return not (self._is_collided(state1)
                        or self._is_collided(state2))
        n_steps = int(np.ceil(dist / self.cd_step_size))
        for i in range(n_steps + 1):
            t = i / n_steps
            s = self.state_space.interpolate(state1, state2, t)
            if self._is_collided(s):
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
        # TODO joint range check?
        collided = self.collider.is_collided(state)
        if len(self._collision_cache) >= self._cache_size:
            self._collision_cache.clear()
        self._collision_cache[key] = collided
        return collided

    def _state_to_key(self, state):
        return tuple(np.round(state, decimals=3))
