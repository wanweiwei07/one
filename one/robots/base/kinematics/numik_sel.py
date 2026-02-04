import os
import time
import numpy as np
import one.utils.math as oum
import one.robots.base.kinematics.numik as orbkin
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
        # CVT data
        self._cvt_q = None
        self._cvt_tf = None
        self._feat = None
        self._tree = None
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
        _, ids = self._tree.query(feat, k=k)
        return self._cvt_q[ids], ids

    def ik(self, root_rotmat, root_pos,
           tgt_rotmat, tgt_pos, max_solutions=8,
           ref_qs=None, max_iter=12, **kwargs):
        if self._tree is None:
            raise RuntimeError("CVT database not loaded.")
        root_tf = oum.tf_from_rotmat_pos(root_rotmat, root_pos)
        tgt_tf = oum.tf_from_rotmat_pos(tgt_rotmat, tgt_pos)
        seeds, ids = self.query_seeds(
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
            ids = ids[sorted_indices]
        sols = []
        for sid, qs0 in zip(ids, seeds):
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
            info["seed_id"] = int(sid)
            sols.append((qs, info))
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
            print(f"[CVT] database loaded from {self._data_dir}")
            return True
        except Exception as e:
            print(f"[CVT] load failed: {e}")
            return False

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
        # tic = time.time()
        for i in range(self.n_cvt):
            tf = self.fk(self._cvt_q[i], root_tf=np.eye(4, dtype=np.float32))
            self._cvt_tf[i] = tf
            self._feat[i, :3] = tf[:3, 3]
            self._feat[i, 3:] = oum.rotvec_from_rotmat(tf[:3, :3])
            # # progress report
            # if (i + 1) % 50 == 0 or (i + 1) == self.n_cvt:
            #     elapsed = time.time() - tic
            #     percent = 100.0 * (i + 1) / self.n_cvt
            #     eta = elapsed / (i + 1) * (self.n_cvt - (i + 1))
            #     print(f"[CVT] FK {i + 1:5d}/{self.n_cvt} "
            #           f"({percent:4.2f}%)  "
            #           f"elapsed: {elapsed:2.1f}s  ETA: {eta:6.1f}s")
        self._tree = cKDTree(self._feat)
        print(f"[CVT] database built: {self.n_cvt} samples")

    def _save_cvt_database(self):
        np.save(self._q_path, self._cvt_q)
        np.save(self._tf_path, self._cvt_tf)
        np.save(self._feat_path, self._feat)
        import pickle
        pickle.dump(self._tree, open(self._tree_path, "wb"))
        print(f"[CVT] database saved to {self._data_dir}")
