"""A multi-arm cell: several arms sharing ONE collision world.

Promotes the bookkeeping every multi-arm workcell otherwise hand-rolls -- build
the shared collider once, and before planning a leg make the moving arm the sole
actor while freezing the others as obstacles. The arms themselves are ordinary
:class:`~one.manipulation.arm.SingleArmManipulation` objects; this only owns the
world they share and the per-leg setup, so a dual-arm assembly recipe stays a
thin sequence of ``cell.activate(arm, aux=...)`` + ``arm.approach/transfer/...``.
"""
import numpy as np

import one.collider.mj_collider as ocm


class Workcell:
    """Several arms (each a ``SingleArmManipulation``) sharing one collider.

    Subclass and populate ``self.arms`` (an ordered ``dict`` ``name -> arm``) in
    ``__init__``; ``build_collider`` / ``activate`` are inherited. Typical use::

        cell.build_collider(fixtures=cell.statics)
        cell.activate(cell.arms['left'], aux={'right': home_r})
        motion = cell.arms['left'].approach(pose, collider=cell.collider,
                                            tcp=..., constraints=[cable_l])
    """

    def __init__(self):
        self.arms = {}        # name -> SingleArmManipulation (subclass fills this)
        self.collider = None

    def build_collider(self, fixtures=(), *, margin=0.0):
        """Build the shared collision world ONCE: every arm (+ its mounted end
        effector, pulled in automatically) + the static ``fixtures``; NO
        manipulated objects -- those are grasped (``arm.grasp``) or excluded, not
        obstacles. ``actors`` spans all arms so the auto-ACM covers the whole
        cell. Stored on ``self.collider`` and returned."""
        c = ocm.MJCollider()
        bodies = [a.body for a in self.arms.values()]
        for b in bodies:
            c.append(b)
        for f in fixtures:
            c.append(f)
        c.actors = bodies
        c.compile(margin=margin, auto_acm=True)
        self.collider = c
        return c

    def activate(self, arm, aux=None):
        """Prepare ``self.collider`` to plan ``arm``: make it the SOLE actor and
        freeze every OTHER arm at ``aux[name]`` (a joint config) as a static
        obstacle. Call before each of ``arm``'s planning verbs; an arm omitted
        from ``aux`` keeps its current pose. Returns ``self.collider`` for
        convenience."""
        if self.collider is None:
            raise RuntimeError("call build_collider() before activate()")
        self.collider.actors = [arm.body]
        aux = aux or {}
        for name, other in self.arms.items():
            if other is arm:
                continue
            qs = aux.get(name)
            if qs is not None:
                self.collider.set_mecba_qpos(
                    other.body, np.asarray(qs, dtype=np.float32))
        return self.collider

    def plan(self, arm, *, aux=None, constraints=(), tcp=None, start_qs=None):
        """``activate(arm, aux)`` then open a manipulation
        :class:`~one.manipulation.session.Session` over the shared collider --
        the one-call entry point for a multi-arm leg: freeze the other arms,
        then list ``moveto`` / ``grasp`` / ``linear`` / ``release`` steps and read
        ``.result``."""
        self.activate(arm, aux)
        return arm.plan(self.collider, constraints=constraints, tcp=tcp,
                        start_qs=start_qs)
