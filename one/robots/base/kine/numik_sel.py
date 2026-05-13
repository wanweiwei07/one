import os
import time
import numpy as np
import one.utils.math as oum
import one.robots.base.kine.numik as orbkin
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2


class SELIKSolver(orbkin.NumIKSolver):

    def __init__(self, chain, data_dir,
                 n_cvt=2048, n_pool=200000, n_iter=20):
        super().__init__(chain)
        self.n_cvt = n_cvt
        self.n_pool = n_pool
        self.n_iter = n_iter
        self._k = 16
        # data directory
        self._data_dir = data_dir
        os.makedirs(self._data_dir, exist_ok=True)
        # file prefix
        self._q_path = os.path.join(self._data_dir, "cvt_q.npy")
        self._tf_path = os.path.join(self._data_dir, "cvt_tf.npy")
        self._feat_path = os.path.join(self._data_dir, "cvt_feat.npy")
        self._tree_path = os.path.join(self._data_dir, "cvt_tree.pkl")
        self._jinv_path = os.path.join(self._data_dir, "cvt_jinv.npy")
        # CVT data
        self._cvt_q = None
        self._cvt_tf = None
        self._feat = None
        self._tree = None
        self._cvt_jinv = None
        # try load or build
        if not self._try_load_database():
            print("[CVT] database not found, building ...")
            self.build_and_save_cvt_database()

    def build_and_save_cvt_database(self):
        print("[CVT] building database ...")
        self._build_cvt_database()
        self._save_cvt_database()

    def query_seeds(self, tgt_tf, k=16):
        p = tgt_tf[:3, 3]
        r = oum.rotvec_from_rotmat(tgt_tf[:3, :3])
        feat = np.concatenate([p, r]).astype(np.float32)
        # Widen the KDTree candidate pool, then rerank by linearized
        # joint-space distance ||J^-1 @ (target - seed_tcp)|| so the first
        # seeds passed to the backbone solver actually land in its basin
        # of attraction. Same number (k) of seeds returned to ik().
        k_max = min(self.n_cvt, max(k * 10, 200))
        _, ids = self._tree.query(feat, k=k_max)
        seed_tcp = self._feat[ids]                          # (k_max, 6)
        seed_jinv = self._cvt_jinv[ids]                     # (k_max, n_dof, 6)
        diff = (feat - seed_tcp).astype(np.float32)         # (k_max, 6)
        dq = np.einsum('ijk,ik->ij', seed_jinv, diff)       # (k_max, n_dof)
        score = np.sum(dq * dq, axis=1)
        order = np.argsort(score)
        return self._cvt_q[ids[order[:k]]]

    def ik(self, root_rotmat, root_pos,
           tgt_rotmat, tgt_pos, max_solutions=8,
           ref_qs=None, max_iter=12, **kwargs):
        if self._tree is None:
            raise RuntimeError("CVT database not loaded.")
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        seeds = self.query_seeds(
            np.linalg.inv(root_tf) @ tgt_tf, k=self._k)
        if ref_qs is not None:
            prefer_qs = np.asarray(ref_qs, dtype=np.float32)
            if prefer_qs.shape[0] != self._chain.n_active_jnts:
                raise ValueError(
                    f"prefer_qs must have {self._chain.n_active_jnts} elements "
                    f"(active joints), got {prefer_qs.shape[0]}")
            distances = np.linalg.norm(
                seeds - prefer_qs[np.newaxis, :], axis=1)
            sorted_indices = np.argsort(distances)
            seeds = seeds[sorted_indices]
        sols = []
        for qs0 in seeds:
            qs, info = self._backward(
                root_rotmat, root_pos, tgt_rotmat, tgt_pos,
                qs_active_init=qs0, max_iter=max_iter)
            if not info.get("converged", False):
                # if info.get("reason", "") == "joint_limits_exceeded":
                #     tmp_lft_arm = robot.lft_arm.clone()
                #     tmp_lft_arm.fk(qs=qs)
                #     tmp_lft_arm.attach_to(base.scene)
                #     print(qs)
                #     base.run()
                continue
            sols.append(qs)
            if len(sols) >= max_solutions:
                break
        return sols

    def _try_load_database(self):
        if (not os.path.exists(self._q_path) or
                not os.path.exists(self._tf_path) or
                not os.path.exists(self._feat_path) or
                not os.path.exists(self._tree_path)):
            return False
        try:
            self._cvt_q = np.load(self._q_path)
            self._cvt_tf = np.load(self._tf_path)
            self._feat = np.load(self._feat_path)
            import pickle
            self._tree = pickle.load(open(self._tree_path, "rb"))
            if os.path.exists(self._jinv_path):
                self._cvt_jinv = np.load(self._jinv_path)
            else:
                # backfill J^-1 cache for legacy DBs without rebuilding CVT
                print("[CVT] cvt_jinv not found, computing from existing centroids ...")
                self._cvt_jinv = self._compute_cvt_jinv(self._cvt_q)
                np.save(self._jinv_path, self._cvt_jinv)
            print(f"[CVT] database loaded from {self._data_dir}")
            return True
        except Exception as e:
            print(f"[CVT] load failed: {e}")
            return False

    def _compute_cvt_jinv(self, centroids):
        n = centroids.shape[0]
        n_act = self._chain.n_active_jnts
        jinv = np.empty((n, n_act, 6), dtype=np.float32)
        eye4 = np.eye(4, dtype=np.float32)
        for i in range(n):
            _, jacmat, _ = self._forward(centroids[i], root_tf=eye4)
            jinv[i] = np.linalg.pinv(jacmat, rcond=1e-4)
        return jinv

    def _build_cvt_database(self):
        l = self._chain.lmt_lo.astype(np.float32)
        u = self._chain.lmt_up.astype(np.float32)
        dim = self._chain.n_active_jnts
        pool = l + np.random.rand(self.n_pool, dim).astype(np.float32) * (u - l)
        init = l + np.random.rand(self.n_cvt, dim).astype(np.float32) * (u - l)
        centroids, _ = kmeans2(pool, init, iter=self.n_iter, minit='matrix')
        self._cvt_q = centroids.astype(np.float32)
        self._cvt_tf = np.empty((self.n_cvt, 4, 4), dtype=np.float32)
        self._feat = np.empty((self.n_cvt, 6), dtype=np.float32)
        self._cvt_jinv = np.empty((self.n_cvt, dim, 6), dtype=np.float32)
        eye4 = np.eye(4, dtype=np.float32)
        for i in range(self.n_cvt):
            _, jacmat, tf = self._forward(self._cvt_q[i], root_tf=eye4)
            self._cvt_tf[i] = tf
            self._feat[i, :3] = tf[:3, 3]
            self._feat[i, 3:] = oum.rotvec_from_rotmat(tf[:3, :3])
            self._cvt_jinv[i] = np.linalg.pinv(jacmat, rcond=1e-4)
        self._tree = cKDTree(self._feat)
        print(f"[CVT] database built: {self.n_cvt} samples")

    def _save_cvt_database(self):
        np.save(self._q_path, self._cvt_q)
        np.save(self._tf_path, self._cvt_tf)
        np.save(self._feat_path, self._feat)
        np.save(self._jinv_path, self._cvt_jinv)
        import pickle
        pickle.dump(self._tree, open(self._tree_path, "wb"))
        print(f"[CVT] database saved to {self._data_dir}")
