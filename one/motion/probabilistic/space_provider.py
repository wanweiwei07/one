import numpy as np
import one.motion.probabilistic.state_space as ssp

class SpaceProvider:

    @classmethod
    def from_box_bounds(cls, lmt_low, lmt_high,
                        is_state_valid=None,
                        max_edge_step=None):
        space = ssp.RealVectorStateSpace(lmt_low, lmt_high)
        return cls(space, is_state_valid, max_edge_step)

    def __init__(self, state_space, is_state_valid=None, max_edge_step=None):
        """is_state_valid: callback function: (state: np.ndarray) -> bool"""
        self.ssp = state_space
        self.is_state_valid = is_state_valid
        if max_edge_step is not None and max_edge_step <= 0.0:
                raise ValueError("max_edge_step must be positive.")
        self.max_edge_step = max_edge_step

    def check_motion(self, state1, state2):
        dist = self.ssp.distance(state1, state2)
        if dist == 0.0:
            return self.is_state_valid(state1)
        if self.max_edge_step is None or dist <= self.max_edge_step:
            return (self.is_state_valid(state1)
                    and self.is_state_valid(state2))
        n_steps = int(np.ceil(dist / self.max_edge_step))
        for i in range(n_steps + 1):
            t = i / n_steps
            s = self.ssp.interpolate(state1, state2, t)
            if not self.is_state_valid(s):
                return False
        return True

    def enforce_bounds(self, s):
        return self.ssp.enforce_bounds(s)

    def states_equal(self, s1, s2, tol=1e-4):
        return self.ssp.distance(s1, s2) <= tol