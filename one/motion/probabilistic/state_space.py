import numpy as np

class RealVectorStateSpace:
    """Simple R^n box-bounded state space, OMPL-style."""

    def __init__(self, lmt_low, lmt_high):
        """lmt_low, lmt_high: array-like of shape (n,)"""
        self._lmt_low = np.array(lmt_low, dtype=np.float32)
        self._lmt_high = np.array(lmt_high, dtype=np.float32)
        assert self._lmt_low.shape == self._lmt_high.shape
        self.dim = self._lmt_low.size

    def sample_uniform(self):
        ratios = np.random.rand(self.dim)
        return self._lmt_low + (self._lmt_high - self._lmt_low) * ratios

    def distance(self, s1, s2):
        return float(np.linalg.norm(s1 - s2))

    def vectorized_distance(self, s1, s2):
        return np.linalg.norm(s1 - s2, axis=-1)

    def interpolate(self, s1, s2, t):
        """Linear interpolation: t in [0, 1]."""
        return (1.0 - t) * s1 + t * s2

    def enforce_bounds(self, s):
        return np.clip(s, self._lmt_low, self._lmt_high)