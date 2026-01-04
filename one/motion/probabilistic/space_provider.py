import numpy as np
import one.motion.probabilistic.state_space as ssp


class SpaceProvider:

    @classmethod
    def from_box_bounds(cls, lmt_low, lmt_high,
                        collider=None,
                        max_edge_step=None):
        space = ssp.RealVectorStateSpace(lmt_low, lmt_high)
        return cls(space, collider, max_edge_step)

    def __init__(self, state_space, collider=None, max_edge_step=None):
        """is_state_valid: callback function: (state: np.ndarray) -> bool"""
        self.ssp = state_space
        self.collider = collider
        if max_edge_step is not None and max_edge_step <= 0.0:
            raise ValueError("max_edge_step must be positive.")
        self.max_edge_step = max_edge_step

    def is_motion_valid(self, state1, state2):
        dist = self.ssp.distance(state1, state2)
        if dist == 0.0:
            return self.collider.is_collided(state1)
        if self.max_edge_step is None or dist <= self.max_edge_step:
            return not (self.collider.is_collided(state1)
                        and self.collider.is_collided(state2))
        n_steps = int(np.ceil(dist / self.max_edge_step))
        for i in range(n_steps + 1):
            t = i / n_steps
            s = self.ssp.interpolate(state1, state2, t)
            if self.collider.is_collided(s):
                return False
        return True

    def enforce_bounds(self, s):
        return self.ssp.enforce_bounds(s)

    def states_equal(self, s1, s2, tol=1e-4):
        return self.ssp.distance(s1, s2) <= tol
