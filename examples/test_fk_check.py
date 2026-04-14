from one.robots.manipulators.kawasaki.rs007l import rs007l
import numpy as np

robot = rs007l.RS007L()
robot.fk(np.array([np.pi/2, 0, 0, 0, 0, 0], dtype=np.float32))

print("Link positions after q1=90°:")
for i in range(7):
    pos = robot.gl_lnk_tfarr[i][:3, 3]
    print(f'Link {i}: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]')
