import cProfile
import numpy as np
from one import ovw, ossop

profiler = cProfile.Profile()
profiler.enable()

base = ovw.World(cam_pos=np.array([1.5, 0, 1]),
                toggle_auto_cam_orbit=True)
for i in range(2000):
    pos = np.random.rand(3)
    cyl = ossop.gen_frame(pos=pos)
    cyl.attach_to(base.scene)
base.run()

profiler.disable()
profiler.dump_stats("result.prof")
print("Profile saved to result.prof")