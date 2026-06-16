"""Plan O6 left-hand antipodal grasps on an ossop.cylinder PRIMITIVE of the same
size as cylinder.stl (dia 0.025, height 0.075, base at z=0). Counterpart to
o6cylstlplanning.py (the mesh) -- run both and compare with o6cmpplanning.py to
check the scene-object primitive and the STL give consistent grasps.

Headless: ONE_HEADLESS=1   Viewer keys: N = next grasp pair
"""
import os

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
from _o6cylplan import plan_save_show, CYL_RADIUS, CYL_HEIGHT, _THIS

OUT_JSON = os.path.join(_THIS, "o6_cyl_scnobj_grasps.json")


def main():
    cyl = ossop.cylinder(
        spos=(0.0, 0.0, 0.0), epos=(0.0, 0.0, CYL_HEIGHT),
        radius=CYL_RADIUS, segments=24,
        collision_type=ouc.CollisionType.MESH, is_free=True,
        rgb=(0.6, 0.7, 0.5))
    plan_save_show(cyl, OUT_JSON, "scnobj")


if __name__ == "__main__":
    main()
