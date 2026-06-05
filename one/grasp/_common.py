"""Shared grasp-domain helpers (depend on end-effector runtime links and
the collider backends, so they live in the grasp package rather than in
the generic geometry/math utilities)."""
import one.collider.cpu_simd as occs
import one.collider.gpu_simd_batch as ocgcb


def build_ee_target_detector(ee, target_sobj):
    """Collision detector + batch checking every link of an end effector
    against a target object. Prefers the GPU backend, falls back to CPU.

    :param ee: end effector exposing ``runtime_lnks``
    :param target_sobj: the object the ee is tested against
    :return: (detector, batch); a placement collides when
        ``detector.detect_collision_batch(batch) is not None``.
    """
    items = ee.runtime_lnks + [target_sobj]
    tgt_idx = len(items) - 1
    pairs = [(i, tgt_idx) for i in range(len(ee.runtime_lnks))]
    try:
        detector = ocgcb.create_detector()
        batch = ocgcb.build_batch(items, pairs)
    except Exception:
        detector = occs.create_detector()
        batch = occs.build_batch(items, pairs)
    return detector, batch
