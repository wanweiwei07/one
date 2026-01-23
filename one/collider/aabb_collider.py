import numpy as np
import one.collider.cpu_simd as occs
import one.collider.collider_base as ocb


class AABBCollider(ocb.ColliderBase):
    """
    OBB/SAT batch collider based on ColliderBase pair compilation.
    """
    def __init__(self):
        super().__init__()
        self._aabb_local_mins = None  # (N, 3)
        self._aabb_local_maxs = None  # (N, 3)
        self._transforms = None # (N, 4, 4)

    def _post_compile(self):
        # Precompute local AABB for each collision object
        mins = []
        maxs = []
        for obj in self._pair_items:
            tris = occs.cols_to_tris(obj.collisions)
            min_c, max_c = occs.compute_aabb(tris)
            mins.append(min_c)
            maxs.append(max_c)
        self._aabb_local_mins = np.array(mins, dtype=np.float32)
        self._aabb_local_maxs = np.array(maxs, dtype=np.float32)
        # Prepare transform buffer
        n_items = len(self._pair_items)
        self._transforms = np.zeros((n_items, 4, 4), dtype=np.float32)

    def is_collided(self, qs):
        if not self._compiled:
            raise RuntimeError('AABBCollider must be compiled!')
        # 1) FK update
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        # 2) Batch transform all AABBs to world space as OBBs
        centers, half_extents, rotmats = self._batch_transform_all_obbs()
        # 3) Vectorized pair checking
        if self._check_pairs is None or len(self._check_pairs) == 0:
            return False
        indices_a = self._check_pairs[:, 0]
        indices_b = self._check_pairs[:, 1]
        centers_a = centers[indices_a]
        half_extents_a = half_extents[indices_a]
        rotmats_a = rotmats[indices_a]
        centers_b = centers[indices_b]
        half_extents_b = half_extents[indices_b]
        rotmats_b = rotmats[indices_b]
        intersects = self._batch_obb_intersect(
            centers_a, half_extents_a, rotmats_a,
            centers_b, half_extents_b, rotmats_b)
        return bool(intersects.any())

    def _batch_transform_all_obbs(self):
        n_items = len(self._pair_items)
        if n_items == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, np.zeros((0, 3, 3), dtype=np.float32)
        for i, obj in enumerate(self._pair_items):
            self._transforms[i] = obj.tf
        local_centers = (self._aabb_local_mins + self._aabb_local_maxs) * 0.5
        half_extents = (self._aabb_local_maxs - self._aabb_local_mins) * 0.5
        rotmats = self._transforms[:, :3, :3]
        positions = self._transforms[:, :3, 3]
        world_centers = np.einsum('nij,nj->ni', rotmats, local_centers) + positions
        return world_centers, half_extents, rotmats

    def _batch_obb_intersect(self, centers_a, half_extents_a, rotmats_a,
                             centers_b, half_extents_b, rotmats_b, eps=1e-6):
        """
        Batch OBB vs OBB collision detection using optimized Separating Axis Theorem (SAT)
        Fully vectorized implementation testing 15 potential separating axes:
        - 3 axes from OBB A (its local x, y, z axes)
        - 3 axes from OBB B (its local x, y, z axes)
        - 9 cross products of axes from A and B
        :param centers_a, centers_b: (M, 3) - OBB centers in world space
        :param half_extents_a, half_extents_b: (M, 3) - OBB half extents (in local frame)
        :param rotmats_a, rotmats_b: (M, 3, 3) - OBB rotation matrices (world frame)
        :param eps: numerical tolerance for parallel axis detection
        :return: (M,) boolean array - True if OBBs intersect
        """
        # Relative center vector (in world space)
        T = centers_b - centers_a  # (M, 3)
        # Precompute rotation matrix for B in A's frame: R = A^T @ B
        # Directly compute A.T @ B using einsum
        R = np.einsum('mji,mjk->mik', rotmats_a, rotmats_b)  # (M, 3, 3)
        R_abs = np.abs(R) + eps  # Add eps to avoid division by zero
        # Translate T into A's frame
        T_in_A = np.einsum('mij,mj->mi', rotmats_a.transpose(0, 2, 1), T)  # (M, 3)
        # Test axes from OBB A (3 axes) - all at once
        ra = half_extents_a  # (M, 3)
        rb = np.einsum('mij,mj->mi', R_abs, half_extents_b)  # (M, 3)
        no_sep_A = np.abs(T_in_A) <= ra + rb + eps  # (M, 3)
        # Test axes from OBB B (3 axes) - all at once
        # Project T onto B's axes, then take absolute value
        T_dot_B = np.einsum('mji,mi->mj', R.transpose(0, 2, 1), T_in_A)  # (M, 3) - FIX: was 'mij,mi->mj'
        ra_on_B = np.einsum('mji,mi->mj', R_abs, half_extents_a)  # (M, 3)
        rb_on_B = half_extents_b  # (M, 3)
        no_sep_B = np.abs(T_dot_B) <= ra_on_B + rb_on_B + eps  # (M, 3)
        # Test cross product axes (9 axes) - fully vectorized
        # Only test candidates that passed the first 6 axes
        # For axis A[i] × B[j], the projection formula simplifies using R
        # IMPORTANT: When A[i] and B[j] are parallel/anti-parallel, the cross product
        # is near-zero and doesn't provide a valid separating axis. We handle this by
        # detecting when the sum ra + rb is very small (indicating degenerate axis)
        # and skipping that axis test.
        # Precompute all 9 cross-axis projections at once
        # Each cross axis test involves indices (i, i1, i2, j) where i1=(i+1)%3, i2=(i+2)%3
        # We can vectorize by explicitly computing all 9 combinations
        # proj_T for all 9 axes: |T[i2]*R[i1,j] - T[i1]*R[i2,j]|
        # Pattern: (i,j) → uses T_in_A[i2], T_in_A[i1], R[i1,j], R[i2,j]
        # where i1=(i+1)%3, i2=(i+2)%3
        proj_T_all = np.abs(np.stack([
            T_in_A[:, 2] * R[:, 1, 0] - T_in_A[:, 1] * R[:, 2, 0],  # A0×B0: i=0,j=0 → i1=1,i2=2
            T_in_A[:, 2] * R[:, 1, 1] - T_in_A[:, 1] * R[:, 2, 1],  # A0×B1
            T_in_A[:, 2] * R[:, 1, 2] - T_in_A[:, 1] * R[:, 2, 2],  # A0×B2
            T_in_A[:, 0] * R[:, 2, 0] - T_in_A[:, 2] * R[:, 0, 0],  # A1×B0: i=1,j=0 → i1=2,i2=0
            T_in_A[:, 0] * R[:, 2, 1] - T_in_A[:, 2] * R[:, 0, 1],  # A1×B1
            T_in_A[:, 0] * R[:, 2, 2] - T_in_A[:, 2] * R[:, 0, 2],  # A1×B2
            T_in_A[:, 1] * R[:, 0, 0] - T_in_A[:, 0] * R[:, 1, 0],  # A2×B0: i=2,j=0 → i1=0,i2=1
            T_in_A[:, 1] * R[:, 0, 1] - T_in_A[:, 0] * R[:, 1, 1],  # A2×B1
            T_in_A[:, 1] * R[:, 0, 2] - T_in_A[:, 0] * R[:, 1, 2],  # A2×B2
        ], axis=1))  # (M, 9)
        # ra_cross for all 9 axes: half_a[i1]*R_abs[i2,j] + half_a[i2]*R_abs[i1,j]
        ra_cross_all = np.stack([
            half_extents_a[:, 1] * R_abs[:, 2, 0] + half_extents_a[:, 2] * R_abs[:, 1, 0],  # A0×B0
            half_extents_a[:, 1] * R_abs[:, 2, 1] + half_extents_a[:, 2] * R_abs[:, 1, 1],  # A0×B1
            half_extents_a[:, 1] * R_abs[:, 2, 2] + half_extents_a[:, 2] * R_abs[:, 1, 2],  # A0×B2
            half_extents_a[:, 2] * R_abs[:, 0, 0] + half_extents_a[:, 0] * R_abs[:, 2, 0],  # A1×B0
            half_extents_a[:, 2] * R_abs[:, 0, 1] + half_extents_a[:, 0] * R_abs[:, 2, 1],  # A1×B1
            half_extents_a[:, 2] * R_abs[:, 0, 2] + half_extents_a[:, 0] * R_abs[:, 2, 2],  # A1×B2
            half_extents_a[:, 0] * R_abs[:, 1, 0] + half_extents_a[:, 1] * R_abs[:, 0, 0],  # A2×B0
            half_extents_a[:, 0] * R_abs[:, 1, 1] + half_extents_a[:, 1] * R_abs[:, 0, 1],  # A2×B1
            half_extents_a[:, 0] * R_abs[:, 1, 2] + half_extents_a[:, 1] * R_abs[:, 0, 2],  # A2×B2
        ], axis=1)  # (M, 9)
        # rb_cross for all 9 axes: half_b[j1]*R_abs[i,j2] + half_b[j2]*R_abs[i,j1]
        # where j1=(j+1)%3, j2=(j+2)%3
        rb_cross_all = np.stack([
            half_extents_b[:, 1] * R_abs[:, 0, 2] + half_extents_b[:, 2] * R_abs[:, 0, 1],  # A0×B0: j=0→j1=1,j2=2
            half_extents_b[:, 2] * R_abs[:, 0, 0] + half_extents_b[:, 0] * R_abs[:, 0, 2],  # A0×B1: j=1→j1=2,j2=0
            half_extents_b[:, 0] * R_abs[:, 0, 1] + half_extents_b[:, 1] * R_abs[:, 0, 0],  # A0×B2: j=2→j1=0,j2=1
            half_extents_b[:, 1] * R_abs[:, 1, 2] + half_extents_b[:, 2] * R_abs[:, 1, 1],  # A1×B0: j=0→j1=1,j2=2
            half_extents_b[:, 2] * R_abs[:, 1, 0] + half_extents_b[:, 0] * R_abs[:, 1, 2],  # A1×B1: j=1→j1=2,j2=0
            half_extents_b[:, 0] * R_abs[:, 1, 1] + half_extents_b[:, 1] * R_abs[:, 1, 0],  # A1×B2: j=2→j1=0,j2=1
            half_extents_b[:, 1] * R_abs[:, 2, 2] + half_extents_b[:, 2] * R_abs[:, 2, 1],  # A2×B0: j=0→j1=1,j2=2
            half_extents_b[:, 2] * R_abs[:, 2, 0] + half_extents_b[:, 0] * R_abs[:, 2, 2],  # A2×B1: j=1→j1=2,j2=0
            half_extents_b[:, 0] * R_abs[:, 2, 1] + half_extents_b[:, 1] * R_abs[:, 2, 0],  # A2×B2: j=2→j1=0,j2=1
        ], axis=1)  # (M, 9)
        # Compute total radius for each cross axis
        radius_sum = ra_cross_all + rb_cross_all  # (M, 9)
        # Detect degenerate (nearly parallel) axes: when radius_sum is very small,
        # the cross product axis is nearly zero and the test is invalid.
        # Use a threshold of 10*eps to be conservative.
        cross_axis_valid = radius_sum > eps * 10  # (M, 9)
        # Test all 9 cross axes at once
        # For invalid (nearly parallel) axes, force no_sep = True to avoid false separation
        no_sep_cross = (proj_T_all <= radius_sum + eps) | ~cross_axis_valid  # (M, 9)
        # Combine all tests: collision if no separating axis found (all 15 axes must show no separation)
        result = no_sep_A.all(axis=1) & no_sep_B.all(axis=1) & no_sep_cross.all(axis=1)
        return result