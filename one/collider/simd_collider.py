import numpy as np
import one.scene.scene as oss
import one.collider.cpu_simd as cpu_simd
import one.collider.gpu_simd as gpu_simd


class SIMDCollider:
    """
    SIMD-based collision detector using FK + runtime_lnks

    Features:
    - GPU-accelerated with CPU fallback
    - Precompiled collision pairs for performance
    - Collision group filtering always enabled
    - Compatible with MJCollider interface

    Usage:
        collider = SIMDCollider(use_gpu=True)
        collider.append(robot)
        collider.append(obstacle)
        collider.actors = [robot]
        collider.compile()
        is_collided = collider.is_collided(qs)
    """

    def __init__(self, use_gpu=True, eps=1e-9):
        """
        Initialize SIMD collider
        :param use_gpu: If True, try GPU acceleration (fallback to CPU on failure)
        :param eps: Numerical tolerance for collision detection
        """
        self.scene = oss.Scene()
        self._actors = ()
        self._actor_qs_slice = {}
        self._use_gpu = use_gpu
        self._eps = eps
        self._compiled = False
        # Precompiled collision pairs (built in compile())
        # Format: (actor, lidx_i, lidx_j, col_i, col_j)
        self._self_collision_pairs = []
        # Format: (actor, lidx, sobj, col_lnk, col_sobj)
        self._actor_sobj_pairs = []
        # Format: (actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o)
        self._actor_mecba_pairs = []
        # Format: (actor_a, lidx_a, actor_b, lidx_b, col_a, col_b)
        self._actor_actor_pairs = []
        # Statistics
        self._stats = {
            'total_checks': 0,
            'self_collision_checks': 0,
            'actor_obstacle_checks': 0,
            'actor_actor_checks': 0,
            'actual_simd_calls': 0,
            'aabb_culled': 0,
            'collisions_found': 0}

    def append(self, entity):
        """Add robot or obstacle to scene"""
        self.scene.add(entity)

    def compile(self):
        """
        Precompile all collision pairs with optimizations:
        1. Filter out empty collision shapes
        2. Apply collision_ignores_idx filtering
        3. Apply collision_group filtering
        4. Cache collision shape references
        """
        if not self.actors:
            raise RuntimeError("SIMDCollider.actors must be set before compile!")
        # 1. Self-collision pairs
        self._self_collision_pairs = []
        for actor in self._actors:
            lnks = actor.runtime_lnks
            n_lnks = len(lnks)
            ignore_pairs = actor.structure.compiled.collision_ignores_idx
            # Prefilter: find links with collision shapes
            valid_indices = [(i, lnks[i].collisions[0])
                             for i in range(n_lnks) if lnks[i].collisions]
            for idx_i, (i, col_i) in enumerate(valid_indices):
                for idx_j in range(idx_i + 1, len(valid_indices)):
                    j, col_j = valid_indices[idx_j]
                    # Check ignore list
                    pair = (min(i, j), max(i, j))
                    if pair in ignore_pairs:
                        continue
                    # Check collision group
                    if not self._should_collide(lnks[i], lnks[j]):
                        continue
                    # Cache: (actor, lidx_i, lidx_j, col_i, col_j)
                    self._self_collision_pairs.append((actor, i, j, col_i, col_j))
        # 2. Actor vs SceneObject pairs
        self._actor_sobj_pairs = []
        for actor in self._actors:
            lnks = actor.runtime_lnks
            valid_links = [(i, lnk, lnk.collisions[0])
                           for i, lnk in enumerate(lnks) if lnk.collisions]
            for sobj in self.scene.sobjs:
                if not sobj.collisions:
                    continue
                col_sobj = sobj.collisions[0]
                for lidx, lnk, col_lnk in valid_links:
                    if not self._should_collide(lnk, sobj):
                        continue
                    # Cache: (actor, lidx, sobj, col_lnk, col_sobj)
                    self._actor_sobj_pairs.append(
                        (actor, lidx, sobj, col_lnk, col_sobj))
        # 3. Actor vs Non-Actor Robot pairs
        self._actor_mecba_pairs = []
        non_actor_robots = [mb for mb in self.scene.mecbas if mb not in self._actors]
        for actor in self._actors:
            actor_valid = [(i, lnk, lnk.collisions[0])
                           for i, lnk in enumerate(actor.runtime_lnks) if lnk.collisions]
            for robot_obs in non_actor_robots:
                obs_valid = [(i, lnk, lnk.collisions[0])
                             for i, lnk in enumerate(robot_obs.runtime_lnks) if lnk.collisions]
                for actor_lidx, actor_lnk, col_a in actor_valid:
                    for obs_lidx, obs_lnk, col_o in obs_valid:
                        if not self._should_collide(actor_lnk, obs_lnk):
                            continue
                        # Cache: (actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o)
                        self._actor_mecba_pairs.append(
                            (actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o))
        # 4. Actor vs Actor pairs
        self._actor_actor_pairs = []
        n_actors = len(self._actors)
        for i in range(n_actors):
            actor_a = self._actors[i]
            valid_a = [(idx, lnk, lnk.collisions[0])
                       for idx, lnk in enumerate(actor_a.runtime_lnks) if lnk.collisions]
            for j in range(i + 1, n_actors):
                actor_b = self._actors[j]
                valid_b = [(idx, lnk, lnk.collisions[0])
                           for idx, lnk in enumerate(actor_b.runtime_lnks) if lnk.collisions]
                for lidx_a, lnk_a, col_a in valid_a:
                    for lidx_b, lnk_b, col_b in valid_b:
                        if not self._should_collide(lnk_a, lnk_b):
                            continue
                        # Cache: (actor_a, lidx_a, actor_b, lidx_b, col_a, col_b)
                        self._actor_actor_pairs.append(
                            (actor_a, lidx_a, actor_b, lidx_b, col_a, col_b))
        self._compiled = True
        # Print compilation statistics
        print(f"SIMDCollider compiled:")
        print(f"  Self-collision pairs:     {len(self._self_collision_pairs)}")
        print(f"  Actor-SceneObject pairs:  {len(self._actor_sobj_pairs)}")
        print(f"  Actor-Robot pairs:        {len(self._actor_mecba_pairs)}")
        print(f"  Actor-Actor pairs:        {len(self._actor_actor_pairs)}")
        total = (len(self._self_collision_pairs) + len(self._actor_sobj_pairs) +
                 len(self._actor_mecba_pairs) + len(self._actor_actor_pairs))
        print(f"  Total pairs to check:     {total}")

    def is_collided(self, qs):
        """
        Check collision for given joint configuration
        :param qs: joint angles for all actors (concatenated)
        :return: True if collision detected, False otherwise
        """
        if not self._compiled:
            raise RuntimeError("SIMDCollider must be compiled!")
        # Update FK for all actors
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        # 1. Self-collision checks
        for actor, lidx_i, lidx_j, col_i, col_j in self._self_collision_pairs:
            self._stats['total_checks'] += 1
            self._stats['self_collision_checks'] += 1
            lnk_i = actor.runtime_lnks[lidx_i]
            lnk_j = actor.runtime_lnks[lidx_j]
            # Use cached collision shapes, only fetch transforms
            if self._check_pair_direct(col_i, lnk_i.tf, col_j, lnk_j.tf) is not None:
                self._stats['collisions_found'] += 1
                return True
        # 2. Actor vs SceneObject checks
        for actor, lidx, sobj, col_lnk, col_sobj in self._actor_sobj_pairs:
            self._stats['total_checks'] += 1
            self._stats['actor_obstacle_checks'] += 1
            lnk = actor.runtime_lnks[lidx]
            if self._check_pair_direct(
                    col_lnk, lnk.tf, col_sobj, sobj.tf) is not None:
                self._stats['collisions_found'] += 1
                return True
        # 3. Actor vs Non-Actor Robot checks
        for actor, actor_lidx, robot_obs, obs_lidx, col_a, col_o in self._actor_mecba_pairs:
            self._stats['total_checks'] += 1
            self._stats['actor_obstacle_checks'] += 1
            actor_lnk = actor.runtime_lnks[actor_lidx]
            obs_lnk = robot_obs.runtime_lnks[obs_lidx]
            if self._check_pair_direct(
                    col_a, actor_lnk.tf, col_o, obs_lnk.tf) is not None:
                self._stats['collisions_found'] += 1
                return True
        # 4. Actor vs Actor checks
        for actor_a, lidx_a, actor_b, lidx_b, col_a, col_b in self._actor_actor_pairs:
            self._stats['total_checks'] += 1
            self._stats['actor_actor_checks'] += 1
            lnk_a = actor_a.runtime_lnks[lidx_a]
            lnk_b = actor_b.runtime_lnks[lidx_b]
            if self._check_pair_direct(col_a, lnk_a.tf, col_b, lnk_b.tf) is not None:
                self._stats['collisions_found'] += 1
                return True
        return False

    def get_slice(self, actor):
        """Get the slice of qs corresponding to this actor"""
        return self._actor_qs_slice.get(actor, None)

    @property
    def actors(self):
        return self._actors

    @actors.setter
    def actors(self, actors):
        """Set which entities are actors (movable robots)"""
        if not actors:
            raise ValueError("SIMDCollider.actors cannot be empty!")
        self._actors = tuple(actors)
        self._rebuild_mapping()

    def _rebuild_mapping(self):
        """Build qs slice mapping for all actors"""
        self._actor_qs_slice.clear()
        offset = 0
        for actor in self._actors:
            if actor not in self.scene.mecbas:
                raise RuntimeError(
                    "All SIMDCollider.actors must be added to the scene!")
            ndof = actor.ndof
            self._actor_qs_slice[actor] = slice(offset, offset + ndof)
            offset += ndof

    def _should_collide(self, obj_a, obj_b):
        """Check if two objects should collide based on collision groups"""
        ga = obj_a.collision_group
        gb = obj_b.collision_group
        aa = obj_a.collision_affinity
        ab = obj_b.collision_affinity
        return bool((aa & gb) and (ab & ga))

    def _transform_aabb(self, min_local, max_local, tf):
        """
        Transform an AABB by a 4x4 transformation matrix
        :param min_local, max_local: (3,) arrays - local-space AABB bounds
        :param tf: (4,4) transformation matrix
        :return: (transformed_min, transformed_max) as (3,) arrays
        """
        # Generate 8 corner points of the AABB
        corners = np.array([
            [min_local[0], min_local[1], min_local[2]],
            [min_local[0], min_local[1], max_local[2]],
            [min_local[0], max_local[1], min_local[2]],
            [min_local[0], max_local[1], max_local[2]],
            [max_local[0], min_local[1], min_local[2]],
            [max_local[0], min_local[1], max_local[2]],
            [max_local[0], max_local[1], min_local[2]],
            [max_local[0], max_local[1], max_local[2]],
        ], dtype=np.float32)
        # Transform all 8 corners to world space
        rotmat = tf[:3, :3]
        pos = tf[:3, 3]
        transformed_corners = (rotmat @ corners.T).T + pos
        # Compute new AABB from transformed corners
        new_min = transformed_corners.min(axis=0)
        new_max = transformed_corners.max(axis=0)
        return new_min, new_max

    def _check_pair_direct(self, col_a, tf_a, col_b, tf_b):
        """
        Perform actual SIMD collision detection with AABB broad phase
        :param col_a, col_b: collision shape objects (already extracted)
        :param tf_a, tf_b: world transform matrices (4x4)
        :return: collision points array or None
        """
        # # AABB Broad Phase
        # min_a, max_a = col_a.aabb
        # min_b, max_b = col_b.aabb
        # if min_a is not None and min_b is not None:
        #     # Transform AABBs to world space (only 8 corners per shape)
        #     min_a, max_a = self._transform_aabb(min_a, max_a, tf_a)
        #     min_b, max_b = self._transform_aabb(min_b, max_b, tf_b)
        #     # Quick AABB intersection test (reuse cpu_simd.aabb_intersect)
        #     if not cpu_simd.aabb_intersect(min_a, max_a, min_b, max_b):
        #         self._stats['aabb_culled'] += 1
        #         return None  # Early exit - skip expensive SIMD call
        # Narrow Phase - Precise SIMD Collision Detection
        self._stats['actual_simd_calls'] += 1
        if self._use_gpu:
            try:
                return gpu_simd.detect_collision(
                    col_a, tf_a, col_b, tf_b, eps=self._eps)
            except Exception:
                # GPU failed, fallback to CPU
                return cpu_simd.detect_collision(
                    col_a, tf_a, col_b, tf_b, eps=self._eps)
        else:
            return cpu_simd.detect_collision(
                col_a, tf_a, col_b, tf_b, eps=self._eps)

    def get_stats(self):
        """Return current statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics counters"""
        for key in self._stats:
            self._stats[key] = 0

    def print_stats(self):
        """Print formatted statistics"""
        print("\n=== SIMDCollider Statistics ===")
        print(f"Total checks:             {self._stats['total_checks']}")
        print(f"  Self-collision:         {self._stats['self_collision_checks']}")
        print(f"  Actor-obstacle:         {self._stats['actor_obstacle_checks']}")
        print(f"  Actor-actor:            {self._stats['actor_actor_checks']}")
        print(f"AABB culled:              {self._stats['aabb_culled']}")
        print(f"Actual SIMD calls:        {self._stats['actual_simd_calls']}")
        print(f"Collisions found:         {self._stats['collisions_found']}")
        if self._stats['total_checks'] > 0:
            cull_rate = 100 * self._stats['aabb_culled'] / self._stats['total_checks']
            simd_rate = 100 * self._stats['actual_simd_calls'] / self._stats['total_checks']
            print(f"AABB cull rate:           {cull_rate:.1f}%")
            print(f"SIMD call ratio:          {simd_rate:.1f}%")