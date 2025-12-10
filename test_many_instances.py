import cProfile
import numpy as np
from one import wd, scn, prims

profiler = cProfile.Profile()
profiler.enable()

scene = scn.Scene()
for i in range(2000):
    pos = np.random.rand(3)
    print(pos)
    cyl = prims.gen_frame(pos=pos)
    scene.add(cyl)
base = wd.World(cam_pos=np.array([.5, 0, 1]),
                toggle_auto_cam_orbit=True)
base.set_scene(scene)
base.run()

profiler.disable()
profiler.dump_stats("result.prof")
print("Profile saved to result.prof")