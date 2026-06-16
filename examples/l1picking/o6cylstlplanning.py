"""Plan O6 left-hand antipodal grasps on the cylinder loaded from cylinder.stl.
Counterpart to o6scnobjplanning.py (an ossop.cylinder primitive of the same
size) -- run both and compare with o6cmpplanning.py.

Headless: ONE_HEADLESS=1   Viewer keys: N = next grasp pair
"""
import os

import one.utils.constant as ouc
import one.scene.scene_object as osso
from _o6cylplan import plan_save_show, _THIS

CYL_STL = os.path.join(_THIS, "cylinder.stl")
OUT_JSON = os.path.join(_THIS, "o6_cyl_stl_grasps.json")


def main():
    cyl = osso.SceneObject.from_file(
        CYL_STL, collision_type=ouc.CollisionType.MESH, is_free=True,
        rgb=(0.6, 0.7, 0.5))
    plan_save_show(cyl, OUT_JSON, "stl")


if __name__ == "__main__":
    main()
