"""Planning constraints: state-validity predicates layered on top of collision.

A :class:`~one.motion.core.planning_context.PlanningContext` treats a config as
valid when it is in-bounds, satisfies every registered ``Constraint``, AND is
collision-free. Constraints capture the task-specific validity a collider can't:
a tether/cable length limit, a keep-upright wrist, keep-out zones, a reach
envelope, etc. They are first-class and composable -- pass a list to the context
-- so neither the planner nor the manipulation verbs need to know what they are.
"""


class Constraint:
    """A state-validity predicate beyond collision.

    ``is_valid(q)`` returns whether the active actor's joint config ``q`` is
    acceptable. The implementation holds whatever it needs (its robot, a held
    object, keep-out geometry) and may FK the robot itself -- the context passes
    only ``q``. Keep it as cheap as the task allows: it is evaluated at every
    sampled node and along every edge, so an expensive constraint slows planning
    (do the cheap sub-checks first and return early).

    Set ``last_fail`` to a short reason string when returning False to aid
    diagnostics; leave it ``None`` on success.
    """

    last_fail = None

    def is_valid(self, q) -> bool:
        raise NotImplementedError
