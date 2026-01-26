import numpy as np
import one.collider.collider_base as ocb
import one.collider.cpu_simd as occs
import one.collider.gpu_simd_batch as ocgs


class SIMDCollider(ocb.ColliderBase):
    def __init__(self, use_gpu=True, eps=1e-9):
        super().__init__()
        self._use_gpu = use_gpu
        self._eps = eps
        self._transforms = None

    def _post_compile(self):
        n_items = len(self._pair_items)
        self._transforms = np.zeros((n_items, 4, 4), dtype=np.float32)

    def is_collided(self, qs):
        if not self._compiled:
            raise RuntimeError('SIMDCollider must be compiled!')
        # FK update
        for actor, sl in self._actor_qs_slice.items():
            actor.fk(qs[sl])
        self._update_transforms()
        # 逐 pair 检测（可替换为 batch GPU）
        pairs = self._check_pairs
        if pairs is None or len(pairs) == 0:
            return False
        for idx_a, idx_b in pairs:
            item_a = self._pair_items[idx_a]
            item_b = self._pair_items[idx_b]
            col_a = item_a.collisions[0] if item_a.collisions else None
            col_b = item_b.collisions[0] if item_b.collisions else None
            if col_a is None or col_b is None:
                continue

            if self._use_gpu:
                try:
                    pts = ocgs.detect_collision(col_a, item_a.tf, col_b, item_b.tf, eps=self._eps)
                except Exception:
                    pts = occs.detect_collision(col_a, item_a.tf, col_b, item_b.tf, eps=self._eps)
            else:
                pts = occs.detect_collision(col_a, item_a.tf, col_b, item_b.tf, eps=self._eps)

            if pts is not None and len(pts) > 0:
                return True

        return False

    def _update_transforms(self):
        if self._transforms is None:
            self._transforms = np.zeros((len(self._pair_items), 4, 4), dtype=np.float32)
        for i, obj in enumerate(self._pair_items):
            self._transforms[i] = obj.tf
